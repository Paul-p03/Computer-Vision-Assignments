import cv2
import matplotlib
from matplotlib import pyplot as plt
from tkinter import *
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os

def button_click(): 
    global display_label, canvas_hist1, img_copy, dogImage
    
    # Save the modified img_copy as a new image file
    new_image_filename = r"C:\Users\ppatu\Computer-Vision-Assignments\Computer-Vision-Assignments\Assignment1\dogImage1.bmp"
    
    # Save the modified image to the new location
    img_copy.save(new_image_filename)  
    print(f"Image saved to {new_image_filename}")
    
    # Delete the original image
    if os.path.exists(dogImage):
        os.remove(dogImage)
        print(f"Original image deleted: {dogImage}")
    
    os.rename(new_image_filename, dogImage)
    print(f"New image renamed to {dogImage}")
    
    root.quit()  

#Function that makes the direct adjustment to the image itself
def scrollBarPos(brightness_val = 0, contrast_val = 0): #Sets both scroll bar positions to 0
    global img_copy, display_label, canvas_hist2

    effect = controller(np.array(img_copy), brightness_val, contrast_val)
    effect_image = ImageTk.PhotoImage(image=Image.fromarray(effect))
    display_label_copy.config(image=effect_image)
    display_label_copy.image = effect_image

    gray_img = Image.fromarray(effect).convert('L')
    gray_data = np.array(gray_img).flatten()

    a.clear()
    a.hist(gray_data, bins = 255, color = 'blue', edgecolor = 'black')
    canvas_hist2.draw()

#Equation to adjust image values when slider position is adjusted
def controller(img_copy, brightness = 127, contrast = 0):
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255)) 
  
    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127)) 

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            max = 255
        else:
            shadow = 0
            max = 255 + brightness

        al_pha = (max - shadow) / 255
        ga_mma = shadow

        cal = cv2.addWeighted(img_copy, al_pha, img_copy, 0, ga_mma)
    else: 
        cal = img_copy

    if contrast != 0:
        alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        gamma = 127 * (1 - alpha)

        # Apply contrast effect
        cal = cv2.addWeighted(cal, alpha, cal, 0, gamma)
    return cal

root = tk.Tk()

#WINDOW format
root.title("Image Adjustment Settings: ")
root.configure(background = "white")
root.geometry("1200x600+100+100")

#Creates frame to hold image
image_frame = tk.Frame(root)
image_frame.pack(side = tk.BOTTOM)

#Creates left frame for histogram input
left_image = tk.Frame(image_frame)
left_image.pack(side=tk.LEFT)

#Creates right frame for histogram input
right_image = tk.Frame(image_frame)
right_image.pack(side = tk.RIGHT)

dogImage = r"C:\Users\ppatu\Computer-Vision-Assignments\Computer-Vision-Assignments\Assignment1\dogImage.bmp"
img = Image.open(dogImage)
resized_img = img.resize((300, 300)) #Resizes image 

bmp_image = ImageTk.PhotoImage(resized_img) #Converts .bmp file into a readable format
img_copy = resized_img.copy()   #Copys image
img_copy_tk = ImageTk.PhotoImage(img_copy)  #Reformats copied image

display_label = tk.Label(root, image= bmp_image)
display_label.place(relx = 0.17, rely = 0.3, anchor = 'w')   #Displays First Dog Image
display_label_copy = tk.Label(root, image= img_copy_tk)
display_label_copy.place(relx = 0.58, rely = 0.3, anchor = 'w')    #Displays Copy Image

#Converts the image into an array structure
gray_img = resized_img.convert('L')
gray_data = np.array(gray_img).flatten()

f = Figure(figsize = (5,2.5), dpi = 100)
a = f.add_subplot()
a.hist(gray_data, bins = 255, color = 'blue', edgecolor = 'black')

#Plots Histogram 1 
canvas_hist1 = FigureCanvasTkAgg(f, master=left_image)
canvas_hist1.get_tk_widget().pack(side = tk.BOTTOM, fill = tk.BOTH, expand = True)

#plots histogram 2
canvas_hist2 = FigureCanvasTkAgg(f, master = right_image)
canvas_hist2.get_tk_widget().pack(side = tk.BOTTOM, fill = tk.BOTH, expand = True)

#Brightness Slider
brightness_slider = tk.Scale(root, from_=0, to=510, orient = 'horizontal',label='Brightness', command=lambda x: scrollBarPos(int(x), contrast_slider.get()), showvalue= 0)
brightness_slider.set(255)
brightness_slider.place(relx = 0.88, rely = 0.32, anchor = 'w')

#Contrast Slider
contrast_slider = tk.Scale(root, from_=0, to=254, orient='horizontal', label='Contrast', command=lambda x: scrollBarPos(brightness_slider.get(), int(x)), showvalue= 0)
contrast_slider.set(127)  
contrast_slider.place(relx = 0.88, rely = 0.4, anchor = 'w')

button = tk.Button(root, text="Save", command=button_click, anchor="center", bd=3, bg="lightgray")
button.place(relx = 0.915, rely = 0.465, anchor = 'w')

root.mainloop()

