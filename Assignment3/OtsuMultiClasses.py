import cv2 as cv 
import numpy as np
from matplotlib import pyplot as plt
from skimage import img_as_ubyte
from skimage.filters import threshold_multiotsu
from matplotlib.colors import ListedColormap

sittingImage = r"C:\Users\ppatu\Computer-Vision-Assignments\Computer-Vision-Assignments\Assignment3\Images\manSitting.png"
standingImage = r"C:\Users\ppatu\Computer-Vision-Assignments\Computer-Vision-Assignments\Assignment3\Images\manStanding.png"
binaryImage = r"C:\Users\ppatu\Computer-Vision-Assignments\Computer-Vision-Assignments\Assignment3\Images\binaryImage.png"

def binaryOtsu(image_paths):
    for idx, image_path in enumerate(image_paths):     # Cycles through each listed image
        image = cv.imread(image_path, cv.IMREAD_COLOR) # Loads the image with color
        blur = cv.cvtColor(image, cv.COLOR_BGR2GRAY)   # Converts to grayscale for Otsu function

               # Gaussian filter allows for smoother segmentation
    
        # multi level classes
        thresholds = threshold_multiotsu(blur, classes=5)
        regions = np.digitize(blur, bins=thresholds)
        output = img_as_ubyte(regions)
        #Color Thresholds
        colorMap = ListedColormap(['red', 'green', 'blue', 'yellow', 'purple'])
        colorOutput = colorMap(output / output.max())

        # Original image
        plt.subplot(len(image_paths), 2, idx * 2 + 1)
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        plt.title(f"Original Image {idx + 1}")
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks

        # Multi class image
        plt.subplot(len(image_paths), 2, idx * 2 + 2)
        plt.imshow(colorOutput)
        plt.title(f"Multi-Class No Gaussian {idx + 1}")
        plt.xticks([]) 
        plt.yticks([])



def meanShift(image_paths):
    for idx, image_path in enumerate(image_paths):

        image = cv.imread(image_path, cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        sp = 20  
        sr = 40  
        # Apply mean shift filtering
        filtered = cv.pyrMeanShiftFiltering(image, sp, sr)

        # Display the original and filtered images
        plt.subplot(len(image_paths), 2, idx * 2 + 1)
        plt.imshow(image)
        plt.title(f"Original Image {idx + 1}")
        plt.xticks([]) 
        plt.yticks([])

        plt.subplot(len(image_paths), 2, idx * 2 + 2)
        plt.imshow(filtered)
        plt.title(f"Multi-Class Segmentation {idx + 1}")
        plt.xticks([]) 
        plt.yticks([])

image_paths = [sittingImage, standingImage, binaryImage]

binaryOtsu(image_paths)
#meanShift(image_paths)
plt.show()
