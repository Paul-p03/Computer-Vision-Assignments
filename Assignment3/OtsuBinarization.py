import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

sittingImage = r"C:\Users\ppatu\Computer-Vision-Assignments\Computer-Vision-Assignments\Assignment3\Images\manSitting.png"
standingImage = r"C:\Users\ppatu\Computer-Vision-Assignments\Computer-Vision-Assignments\Assignment3\Images\manStanding.png"
binaryImage = r"C:\Users\ppatu\Computer-Vision-Assignments\Computer-Vision-Assignments\Assignment3\Images\binaryImage.png"

def binaryOtsu(image_paths):
    for idx, image_path in enumerate(image_paths):     #Cycles through each listed image
        
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE) #Loads the image with color
        _, th = cv.threshold(image, 0,255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        plt.subplot(len(image_paths), 2, idx * 2 + 1)
        plt.imshow(image, cmap = 'gray')
        plt.title(f"Original Image {idx + 1}")
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks

        plt.subplot(len(image_paths), 2, idx * 2 + 2)
        plt.imshow(th, cmap = 'gray')
        plt.title(f"Overlayed Image {idx + 1}")
        plt.xticks([]) 
        plt.yticks([])  


image_paths = [sittingImage, standingImage, binaryImage]

binaryOtsu(image_paths)

plt.show()

