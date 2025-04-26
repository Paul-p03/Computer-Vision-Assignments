import cv2 as cv

path = r"C:\Users\ppatu\Computer-Vision-Assignments\Computer-Vision-Assignments\FinalProject\images\LiscensePlate.jpg"
img = cv.imread(path, cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (500,500))
newImg = cv.medianBlur(img, 5)

sobel = cv.Sobel(newImg, cv.CV_64F, 1, 0, ksize=5)

#creates masked image into 16bit BGR grayscale 
gray_16bit = cv.normalize(sobel, None, 0, 65535, cv.NORM_MINMAX, cv.CV_16U)
gray_16bit_color = cv.cvtColor(gray_16bit, cv.COLOR_GRAY2BGR)

#Creates original image into 16bit BGR grayscale 
gray_16bit_original = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
gray_16bit_original_color = cv.normalize(gray_16bit_original, None, 0, 65535, cv.NORM_MINMAX, cv.CV_16U)

#Displays both two 16bit images
together = cv.hconcat([gray_16bit_original_color, gray_16bit_color])
cv.imshow("Image", together)


cv.waitKey(0)
cv.destroyAllWindows()
