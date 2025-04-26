import cv2 as cv
import numpy as np
import pytesseract


path = r"C:\Users\ppatu\Computer-Vision-Assignments\Computer-Vision-Assignments\FinalProject\images\LiscensePlate.jpg"
img = cv.imread(path, cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (500,500))
newImg = cv.medianBlur(img, 11)

#improves the contrast of the image
clahe = cv.createCLAHE(clipLimit=2)
claheEnhance = np.clip(clahe.apply(newImg) + 70, 170, 255).astype(np.uint8)

#edge detection
sobelx = cv.Sobel(claheEnhance, cv.CV_64F, 1, 0, ksize=5)
sobely = cv.Sobel(claheEnhance, cv.CV_64F, 0, 1, ksize = 5)

magSobel = np.sqrt(sobelx**2+sobely**2)

#separates the image characters into noticable letters
th1 = cv.adaptiveThreshold(magSobel.astype(np.uint8), 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

#creates masked image into 16bit BGR grayscale 
gray_16bit = cv.normalize(th1, None, 0, 65535, cv.NORM_MINMAX, cv.CV_16U)
gray_16bit_color = cv.cvtColor(gray_16bit, cv.COLOR_GRAY2BGR)

#Creates original image into 16bit BGR grayscale 
gray_16bit_original = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
gray_16bit_original_color = cv.normalize(gray_16bit_original, None, 0, 65535, cv.NORM_MINMAX, cv.CV_16U)

#Displays both two 16bit images
together = cv.hconcat([gray_16bit_original_color, gray_16bit_color])
cv.imshow("Image", together)

#OCR method needs to be tweaked
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update this path to your Tesseract installation
text = pytesseract.image_to_string(th1)

print("Extracted Text:")
print(text)

cv.waitKey(0)
cv.destroyAllWindows()
