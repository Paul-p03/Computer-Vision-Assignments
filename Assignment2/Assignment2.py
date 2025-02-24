import cv2
import numpy as np
import tkinter as tk
from tkinter import Frame
from PIL import Image, ImageTk

# Initialize main window
root = tk.Tk()
root.title("Image Adjustment Settings: ")
root.configure(background="white")
root.geometry("1400x750+20+20")

# Create a frame
frame = Frame(root, bg="white")
frame.pack(pady=20)

# Function to add images
def add_image(image_path, row, col, width=200, height=200, apply_filter = False, k = 3):
    if isinstance(image_path, str):
        if apply_filter:
            #Displays filter over image
            filtered_image = boxFilter(image_path, k)
            image = Image.fromarray(filtered_image)
        else:
            #Displays original image
            image = Image.open(image_path)
    else:
        image_matrix = image_path
        image_matrix = cv2.normalize(image_matrix, None, 0, 255, cv2.NORM_MINMAX)
        image_matrix = np.uint8(image_matrix)
        image = Image.fromarray(image_matrix)
    
    image = image.resize((width, height))
    convert_image = ImageTk.PhotoImage(image)
    
    label = tk.Label(frame, image=convert_image, bg="white")
    label.image = convert_image 
    
    # Position image using grid
    label.grid(row=row, column=col, padx=30, pady=10)

# Image paths
dog_image = r"C:\Users\ppatu\Computer-Vision-Assignments\Computer-Vision-Assignments\Assignment2\Images\dog.bmp"
bike_image = r"C:\Users\ppatu\Computer-Vision-Assignments\Computer-Vision-Assignments\Assignment2\Images\bicycle.bmp"

def boxFilter(image_path, k): # Box filter algorithm
    image_cv2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    copy_matrix = np.copy(image_cv2)
    empty_matrix = np.zeros_like(copy_matrix)
    mask = (1/(k**2))*np.ones((k, k))

    rows, cols = copy_matrix.shape

    for i in range(rows - k + 1):
        for j in range(cols - k + 1):
            sub_matrix = copy_matrix[i:i+k, j:j+k]
            filter = np.sum(sub_matrix * mask)
            empty_matrix[i + k//2, j + k//2] = filter

    return empty_matrix

#Box filter built in function
def box_filter_command(image_path, k): 
    image_cv2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred_image = cv2.blur(image_cv2,(k,k))
    return blurred_image

#Sobel filter using cv2 function
def sobel_filter(image_path, k):
    image_cv2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sobelx = cv2.Sobel(image_cv2, cv2.CV_64F, 1, 0, k)
    sobely = cv2.Sobel(image_cv2, cv2.CV_64F, 0, 1, k)
    sobelxy = cv2.magnitude(sobelx, sobely)
    return sobelxy

#sobel filter manual conversion
def sobel_filter_x_command(image_path):
    image_cv2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    kernelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    copy_matrix = image_cv2
    empty_matrix = np.zeros_like(copy_matrix)
    width, height = image_cv2.shape

    #algorithm for convoluting each pixel
    for i in range(1, width - 1):
        for j in range(1, height - 1):
            sub_matrix = copy_matrix[i-1:i+2, j-1:j+2] #Extracts a 3x3 matrix from the image
            convolution = np.sum(kernelx*sub_matrix)   #Convolution of each pixel
            empty_matrix[i][j] = convolution           #Stores the convoluted results in empty matrix

    return empty_matrix

#Gaussian filter using cv2 function
def gaussian_Filter(image_path, k):
    image_cv2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_cv2 = cv2.GaussianBlur(image_cv2, (k, k), 0)
    return image_cv2


#Box filter algorithm
add_image(bike_image, 1, 1, 200, 200, k=3)
add_image(bike_image, 1, 2, 200, 200, apply_filter = True, k=3)
add_image(bike_image, 1, 3, 200, 200, apply_filter = True, k=5)

#cv2 Box filter
cv2_blur = box_filter_command(bike_image, 5)
add_image(dog_image, 2, 2, 200, 200)
add_image(cv2_blur, 2, 3, 200, 200)

#cv2 sobel box filter 
cv2_sobel = sobel_filter(dog_image, 5)
add_image(cv2_sobel, 3, 4, 200, 200)

#Sobel X axis box filter algorithm
sobelx = sobel_filter_x_command(dog_image)
add_image(sobelx, 3, 0, 200, 200)
sobelx = sobel_filter_x_command(bike_image)
add_image(sobelx, 3, 1, 200, 200)

#Gaussian blur filter
cv2_Gaussian = gaussian_Filter(dog_image, 5)
add_image(cv2_Gaussian, 4, 2, 200, 200)
cv2_Gaussian = gaussian_Filter(dog_image, 3)
add_image(cv2_Gaussian, 4, 1, 200, 200)

root.mainloop()
