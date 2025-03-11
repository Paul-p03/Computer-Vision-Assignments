import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

sittingImage = r"C:\Users\ppatu\Computer-Vision-Assignments\Computer-Vision-Assignments\Assignment3\Images\manSitting.png"
standingImage = r"C:\Users\ppatu\Computer-Vision-Assignments\Computer-Vision-Assignments\Assignment3\Images\manStanding.png"
binaryImage = r"C:\Users\ppatu\Computer-Vision-Assignments\Computer-Vision-Assignments\Assignment3\Images\binaryImage.png"

def binaryOtsu(image_paths):
    for idx, image_path in enumerate(image_paths):     #Cycles through each listed image
        image = cv.imread(image_path, cv.IMREAD_COLOR) #Loads the image with color
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)   #Converts to grayscale for Otsu function

        _, th = cv.threshold(gray, 0,255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        channel = cv.merge((th, th, th))    #Creates a holding for each RGB parameter

        combine = cv.bitwise_and(image, channel) #Overlays mask onto image

        #Converts to matplotlib format RGB
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB) 
        overlayImage = cv.cvtColor(combine, cv.COLOR_BGR2RGB)

        plt.subplot(len(image_paths), 2, idx * 2 + 1)
        plt.imshow(image_rgb)
        plt.title(f"Original Image {idx + 1}")
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks

        plt.subplot(len(image_paths), 2, idx * 2 + 2)
        plt.imshow(overlayImage)
        plt.title(f"Overlayed Image {idx + 1}")
        plt.xticks([]) 
        plt.yticks([])  


image_paths = [sittingImage, standingImage, binaryImage]

binaryOtsu(image_paths)

plt.show()

