#
#     # img1 = cv2.imread("imageToSave.png")
# import cv2
#
# originalmage = cv2.imread("./data.jpg")
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
# cv2.imshow("fafa",cartoonImage)
# cv2.waitKey(0)
# # cv2.imwrite('imageToSave.png', cv2.cvtColor(cartoonImage, cv2.COLOR_RGB2BGR))
import cv2
import numpy as np
img1 = cv2.imread("data.jpg")
img1g = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1g = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
img1b = cv2.medianBlur(img1g, 7)
edges = cv2.adaptiveThreshold(img1b, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 5)
cv2.imshow("fafa", edges)
cv2.waitKey(0)
imgf = np.float32(img1).reshape(-1, 3)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
compactness, label, center = cv2.kmeans(imgf, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
final_img = center[label.flatten()]
final_img = final_img.reshape(img1.shape)
final = cv2.bitwise_or(final_img, img1, mask=edges)
# cv2.imshow("fafa", cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
cv2.imshow("fafa", cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)