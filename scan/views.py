import base64
# Create your views here.
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView
import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf

import tf_slim as slim

from tqdm import tqdm


def adaptive_instance_norm(content, style, epsilon=1e-5):
    c_mean, c_var = tf.nn.moments(content, axes=[1, 2], keep_dims=True)
    s_mean, s_var = tf.nn.moments(style, axes=[1, 2], keep_dims=True)
    c_std, s_std = tf.sqrt(c_var + epsilon), tf.sqrt(s_var + epsilon)

    return s_std * (content - c_mean) / c_std + s_mean


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]],
                        initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def conv_spectral_norm(x, channel, k_size, stride=1, name='conv_snorm'):
    with tf.variable_scope(name):
        w = tf.get_variable("kernel", shape=[k_size[0], k_size[1], x.get_shape()[-1], channel])
        b = tf.get_variable("bias", [channel], initializer=tf.constant_initializer(0.0))

        x = tf.nn.conv2d(input=x, filter=spectral_norm(w), strides=[1, stride, stride, 1], padding='SAME') + b

        return x


def self_attention(inputs, name='attention', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        h, w = tf.shape(inputs)[1], tf.shape(inputs)[2]
        bs, _, _, ch = inputs.get_shape().as_list()
        f = slim.convolution2d(inputs, ch // 8, [1, 1], activation_fn=None)
        g = slim.convolution2d(inputs, ch // 8, [1, 1], activation_fn=None)
        s = slim.convolution2d(inputs, 1, [1, 1], activation_fn=None)
        f_flatten = tf.reshape(f, shape=[f.shape[0], -1, f.shape[-1]])
        g_flatten = tf.reshape(g, shape=[g.shape[0], -1, g.shape[-1]])
        beta = tf.matmul(f_flatten, g_flatten, transpose_b=True)
        beta = tf.nn.softmax(beta)

        s_flatten = tf.reshape(s, shape=[s.shape[0], -1, s.shape[-1]])
        att_map = tf.matmul(beta, s_flatten)
        att_map = tf.reshape(att_map, shape=[bs, h, w, 1])
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        output = att_map * gamma + inputs

        return att_map, output


def tf_box_filter(x, r):
    ch = x.get_shape().as_list()[-1]
    weight = 1 / ((2 * r + 1) ** 2)
    box_kernel = weight * np.ones((2 * r + 1, 2 * r + 1, ch, 1))
    box_kernel = np.array(box_kernel).astype(np.float32)
    output = tf.nn.depthwise_conv2d(x, box_kernel, [1, 1, 1, 1], 'SAME')
    return output


def guided_filter(x, y, r, eps=1e-2):
    x_shape = tf.shape(x)
    # y_shape = tf.shape(y)

    N = tf_box_filter(tf.ones((1, x_shape[1], x_shape[2], 1), dtype=x.dtype), r)

    mean_x = tf_box_filter(x, r) / N
    mean_y = tf_box_filter(y, r) / N
    cov_xy = tf_box_filter(x * y, r) / N - mean_x * mean_y
    var_x = tf_box_filter(x * x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = tf_box_filter(A, r) / N
    mean_b = tf_box_filter(b, r) / N

    output = mean_A * x + mean_b

    return output


def resblock(inputs, out_channel=32, name='resblock'):
    with tf.variable_scope(name):
        x = slim.convolution2d(inputs, out_channel, [3, 3],
                               activation_fn=None, scope='conv1')
        x = tf.nn.leaky_relu(x)
        x = slim.convolution2d(x, out_channel, [3, 3],
                               activation_fn=None, scope='conv2')

        return x + inputs


def generator(inputs, channel=32, num_blocks=4, name='generator', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        x = slim.convolution2d(inputs, channel, [7, 7], activation_fn=None)
        x = tf.nn.leaky_relu(x)

        x = slim.convolution2d(x, channel * 2, [3, 3], stride=2, activation_fn=None)
        x = slim.convolution2d(x, channel * 2, [3, 3], activation_fn=None)
        x = tf.nn.leaky_relu(x)

        x = slim.convolution2d(x, channel * 4, [3, 3], stride=2, activation_fn=None)
        x = slim.convolution2d(x, channel * 4, [3, 3], activation_fn=None)
        x = tf.nn.leaky_relu(x)

        for idx in range(num_blocks):
            x = resblock(x, out_channel=channel * 4, name='block_{}'.format(idx))

        x = slim.conv2d_transpose(x, channel * 2, [3, 3], stride=2, activation_fn=None)
        x = slim.convolution2d(x, channel * 2, [3, 3], activation_fn=None)

        x = tf.nn.leaky_relu(x)

        x = slim.conv2d_transpose(x, channel, [3, 3], stride=2, activation_fn=None)
        x = slim.convolution2d(x, channel, [3, 3], activation_fn=None)
        x = tf.nn.leaky_relu(x)

        x = slim.convolution2d(x, 3, [7, 7], activation_fn=None)
        # x = tf.clip_by_value(x, -0.999999, 0.999999)

        return x


def unet_generator(inputs, channel=32, num_blocks=4, name='generator', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        x0 = slim.convolution2d(inputs, channel, [7, 7], activation_fn=None)
        x0 = tf.nn.leaky_relu(x0)

        x1 = slim.convolution2d(x0, channel, [3, 3], stride=2, activation_fn=None)
        x1 = tf.nn.leaky_relu(x1)
        x1 = slim.convolution2d(x1, channel * 2, [3, 3], activation_fn=None)
        x1 = tf.nn.leaky_relu(x1)

        x2 = slim.convolution2d(x1, channel * 2, [3, 3], stride=2, activation_fn=None)
        x2 = tf.nn.leaky_relu(x2)
        x2 = slim.convolution2d(x2, channel * 4, [3, 3], activation_fn=None)
        x2 = tf.nn.leaky_relu(x2)

        for idx in range(num_blocks):
            x2 = resblock(x2, out_channel=channel * 4, name='block_{}'.format(idx))

        x2 = slim.convolution2d(x2, channel * 2, [3, 3], activation_fn=None)
        x2 = tf.nn.leaky_relu(x2)

        h1, w1 = tf.shape(x2)[1], tf.shape(x2)[2]
        x3 = tf.image.resize_bilinear(x2, (h1 * 2, w1 * 2))
        x3 = slim.convolution2d(x3 + x1, channel * 2, [3, 3], activation_fn=None)
        x3 = tf.nn.leaky_relu(x3)
        x3 = slim.convolution2d(x3, channel, [3, 3], activation_fn=None)
        x3 = tf.nn.leaky_relu(x3)

        h2, w2 = tf.shape(x3)[1], tf.shape(x3)[2]
        x4 = tf.image.resize_bilinear(x3, (h2 * 2, w2 * 2))
        x4 = slim.convolution2d(x4 + x0, channel, [3, 3], activation_fn=None)
        x4 = tf.nn.leaky_relu(x4)
        x4 = slim.convolution2d(x4, 3, [7, 7], activation_fn=None)
        # x4 = tf.clip_by_value(x4, -1, 1)
        return x4


def disc_bn(x, scale=1, channel=32, is_training=True,
            name='discriminator', patch=True, reuse=False):
    with tf.variable_scope(name, reuse=reuse):

        for idx in range(3):
            x = slim.convolution2d(x, channel * 2 ** idx, [3, 3], stride=2, activation_fn=None)
            x = slim.batch_norm(x, is_training=is_training, center=True, scale=True)
            x = tf.nn.leaky_relu(x)

            x = slim.convolution2d(x, channel * 2 ** idx, [3, 3], activation_fn=None)
            x = slim.batch_norm(x, is_training=is_training, center=True, scale=True)
            x = tf.nn.leaky_relu(x)

        if patch == True:
            x = slim.convolution2d(x, 1, [1, 1], activation_fn=None)
        else:
            x = tf.reduce_mean(x, axis=[1, 2])
            x = slim.fully_connected(x, 1, activation_fn=None)

        return x


def disc_sn(x, scale=1, channel=32, patch=True, name='discriminator', reuse=False):
    with tf.variable_scope(name, reuse=reuse):

        for idx in range(3):
            x = conv_spectral_norm(x, channel * 2 ** idx, [3, 3],
                                          stride=2, name='conv{}_1'.format(idx))
            x = tf.nn.leaky_relu(x)

            x = conv_spectral_norm(x, channel * 2 ** idx, [3, 3],
                                          name='conv{}_2'.format(idx))
            x = tf.nn.leaky_relu(x)

        if patch == True:
            x = conv_spectral_norm(x, 1, [1, 1], name='conv_out'.format(idx))

        else:
            x = tf.reduce_mean(x, axis=[1, 2])
            x = slim.fully_connected(x, 1, activation_fn=None)

        return x


def disc_ln(x, channel=32, is_training=True, name='discriminator', patch=True, reuse=False):
    with tf.variable_scope(name, reuse=reuse):

        for idx in range(3):
            x = slim.convolution2d(x, channel * 2 ** idx, [3, 3], stride=2, activation_fn=None)
            x = tf.contrib.layers.layer_norm(x)
            x = tf.nn.leaky_relu(x)

            x = slim.convolution2d(x, channel * 2 ** idx, [3, 3], activation_fn=None)
            x = tf.contrib.layers.layer_norm(x)
            x = tf.nn.leaky_relu(x)

        if patch == True:
            x = slim.convolution2d(x, 1, [1, 1], activation_fn=None)
        else:
            x = tf.reduce_mean(x, axis=[1, 2])
            x = slim.fully_connected(x, 1, activation_fn=None)

        return x


def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720 * h / w), 720
        else:
            h, w = 720, int(720 * w / h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image


def cartoonize(load_folder, save_folder, model_path):
    tf.disable_eager_execution()
    input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
    network_out = unet_generator(input_photo)
    final_out = guided_filter(input_photo, network_out, r=1, eps=5e-3)

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.train.Saver(var_list=gene_vars)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    name_list = os.listdir(load_folder)
    for name in tqdm(name_list):
        try:
            load_path = os.path.join(load_folder, name)
            save_path = os.path.join(save_folder, name)
            image = cv2.imread(load_path)
            image = resize_crop(image)
            batch_image = image.astype(np.float32) / 127.5 - 1
            batch_image = np.expand_dims(batch_image, axis=0)
            output = sess.run(final_out, feed_dict={input_photo: batch_image})
            output = (np.squeeze(output) + 1) * 127.5
            output = np.clip(output, 0, 255).astype(np.uint8)
            print(output.shape)
            cv2.imwrite(save_path, output)
        except:
            print('cartoonize {} failed'.format(load_path))
class renderCartoonizeImage(APIView):
    permission_classes = [AllowAny, ]

    def post(self, request):
        model_path = 'saved_models'
        load_folder = 'test_images'
        save_folder = 'cartoonized_images'
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        cartoonize(load_folder, save_folder, model_path)
        # with open("imageToSave.png", "wb") as fh:
        #     fh.write(base64.urlsafe_b64decode(request.data['photo']))
        # originalmage = cv2.imread("imageToSave.png")
        # originalmage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2RGB)
        # # check if the image is chosen
        # grayScaleImage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2GRAY)
        # # applying median blur to smoothen an image
        # smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)
        # # retrieving the edges for cartoon effect
        # getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255,
        #                                 cv2.ADAPTIVE_THRESH_MEAN_C,
        #                                 cv2.THRESH_BINARY, 9, 9)
        # # applying bilateral filter to remove noise
        # # and keep edge sharp as required
        # colorImage = cv2.bilateralFilter(originalmage, 9, 300, 300)
        #
        # # masking edged image with our "BEAUTIFY" image
        # cartoonImage = cv2.bitwise_and(colorImage, colorImage, mask=getEdge)
        # cv2.imwrite('imageToSave.png', cv2.cvtColor(cartoonImage, cv2.COLOR_RGB2BGR))
        return Response({"fafa":"fafaf"},status=status.HTTP_200_OK)
