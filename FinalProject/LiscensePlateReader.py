import cv2 as cv
import numpy as np
import pytesseract
import re

# Setup Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

path = r"C:\Users\ppatu\Computer-Vision-Assignments\Computer-Vision-Assignments\FinalProject\images\LiscensePlate4.jpg"
img = cv.imread(path)
img = cv.resize(img, (1000, 500))

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
gray = cv.resize(gray, None, fx=1.2, fy=1.2, interpolation=cv.INTER_CUBIC)
blur = cv.GaussianBlur(gray, (5,5), 0)
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)

# Morphological operations
rect_kern = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
dilation = cv.dilate(thresh, rect_kern, iterations=1)

# Find contours
contours, hierarchy = cv.findContours(dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
sorted_contours = sorted(contours, key=lambda ctr: cv.boundingRect(ctr)[0])

plate_num = ""

plate_h, plate_w = dilation.shape

# size thresholds (tweak these until they match your plate)
min_h, max_h = int(plate_h*0.5), int(plate_h*0.9)
min_w, max_w = int(plate_w*0.06), int(plate_w*0.12)

for cnt in sorted_contours:
    x, y, w, h = cv.boundingRect(cnt)

    # 1) absolute size filter
    if not (min_h < h < max_h and min_w < w < max_w):
        continue

    # 2) aspect ratio filter (optional)
    ratio = h / float(w)
    if ratio < 1.5 or ratio > 10:
        continue

    # 3) dynamic margin
    m = int(h * 0.2)
    x1, y1 = max(0, x - m), max(0, y - m)
    x2 = min(plate_w, x + w + m)
    y2 = min(plate_h, y + h + m)

    # 4) draw the tightened rectangle
    cv.rectangle(dilation, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ROI extraction
    roi = thresh[y1:y2, x1:x2]
    
    roi = cv.bitwise_not(roi)
    roi = cv.medianBlur(roi, 5)

    text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
    clean_text = re.sub('[\W_]+', '', text)
    plate_num += clean_text


print("License Plate Number:", plate_num)

#cv.imshow("Cropped Plate", box)
cv.imshow('Detected Letters', dilation)
cv.waitKey(0)
cv.destroyAllWindows()
