
    # img1 = cv2.imread("imageToSave.png")
import cv2

originalmage = cv2.imread("data.jpg")
originalmage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2RGB)
# check if the image is chosen
grayScaleImage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2GRAY)
# applying median blur to smoothen an image
smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)
# retrieving the edges for cartoon effect
getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255,
                                cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 9, 9)
# applying bilateral filter to remove noise
# and keep edge sharp as required
colorImage = cv2.bilateralFilter(originalmage, 9, 300, 300)

# masking edged image with our "BEAUTIFY" image
cartoonImage = cv2.bitwise_and(colorImage, colorImage, mask=getEdge)
cv2.imshow("fafa",cartoonImage)
# cv2.imwrite('imageToSave.png', cv2.cvtColor(cartoonImage, cv2.COLOR_RGB2BGR))