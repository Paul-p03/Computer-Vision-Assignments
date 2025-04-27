import cv2 as cv
import numpy as np
import pytesseract
import re

# Setup Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

path = r"C:\Users\ppatu\Computer-Vision-Assignments\Computer-Vision-Assignments\FinalProject\images\LiscensePlate.jpg"
img = cv.imread(path)
img = cv.resize(img, (500, 500))

# Manual select the ROI box
bbox = cv.selectROI("Select License Plate", img, False)
xmin, ymin, w, h = bbox
xmax = xmin + w
ymax = ymin + h
coords = [ymin, ymax, xmin, xmax]
cv.destroyWindow("Select License Plate")

# Crop out the license plate
box = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]

# Preprocessing
gray = cv.cvtColor(box, cv.COLOR_BGR2GRAY)
gray = cv.resize(gray, None, fx=3, fy=3, interpolation=cv.INTER_CUBIC)
blur = cv.GaussianBlur(gray, (5,5), 0)
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)

# Morphological operations
rect_kern = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
dilation = cv.dilate(thresh, rect_kern, iterations=1)

# Find contours
contours, hierarchy = cv.findContours(dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
sorted_contours = sorted(contours, key=lambda ctr: cv.boundingRect(ctr)[0])

plate_num = ""

for cnt in sorted_contours:
    x, y, w, h = cv.boundingRect(cnt)
    height, width = thresh.shape

    if height / float(h) > 6: continue
    ratio = h / float(w)
    if ratio < 1.5: continue
    if width / float(w) > 15: continue
    area = h * w
    if area < 100: continue

    # Expand boundary
    margin = 5
    roi = thresh[max(0, y-margin):min(y+h+margin, height), 
                 max(0, x-margin):min(x+w+margin, width)]
    
    roi = cv.bitwise_not(roi)
    roi = cv.medianBlur(roi, 5)

    text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
    clean_text = re.sub('[\W_]+', '', text)
    plate_num += clean_text


print("License Plate Number:", plate_num)

cv.imshow("Cropped Plate", box)
cv.waitKey(0)
cv.destroyAllWindows()
