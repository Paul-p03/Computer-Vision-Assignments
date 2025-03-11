import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

sittingImage = r"C:\Users\ppatu\Computer-Vision-Assignments\Computer-Vision-Assignments\Assignment3\Images\manSitting.png"
standingImage = r"C:\Users\ppatu\Computer-Vision-Assignments\Computer-Vision-Assignments\Assignment3\Images\manStanding.png"

image1 = cv.imread(sittingImage, cv.IMREAD_GRAYSCALE)
image2 = cv.imread(standingImage, cv.IMREAD_GRAYSCALE)

ret1, th1 = cv.threshold(image1, 127,255, cv.THRESH_BINARY + cv.THRESH_OTSU)
blur = cv.GaussianBlur(image1, (5,5), 0)
ret2, th2 = cv.threshold(blur, 127,255, cv.THRESH_BINARY + cv.THRESH_OTSU)

images = [image1, th1, image1, th2]
titles = ['Original Image 1', 'Otsu Threshold 1', 'Original Image 2', 'Otsu Threshold 2']

for i in range(len(images)):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])  # Add corresponding title
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks

plt.show()

