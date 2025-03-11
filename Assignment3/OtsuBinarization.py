import cv2 as cv
import tkinter as tk
from tkinter import Frame
from PIL import Image, ImageTk

root = tk.Tk()
root.title("Image Segmentation")
root.configure(background = "white")
root.geometry("500x500+475+150")

frame = Frame(root, bg = "black")
frame.pack(pady = 10)

sittingImage = r"C:\Users\ppatu\Computer-Vision-Assignments\Computer-Vision-Assignments\Assignment3\Images\manSitting.png"
standingImage = r"C:\Users\ppatu\Computer-Vision-Assignments\Computer-Vision-Assignments\Assignment3\Images\manStanding.png"

image1 = cv.imread(sittingImage, cv.IMREAD_GRAYSCALE)
image2 = cv.imread(standingImage, cv.IMREAD_GRAYSCALE)
image1_PIL = Image.fromarray(image1)
image1_PIL = image1_PIL.resize((250, 250))
photo = ImageTk.PhotoImage(image1_PIL)

label = tk.Label(frame, image=photo, bg="white")
label.image = photo
label.pack()


root.mainloop()


