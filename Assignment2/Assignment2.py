import tkinter as tk
from tkinter import Frame
from PIL import Image, ImageTk

# Initialize main window
root = tk.Tk()
root.title("Image Adjustment Settings: ")
root.configure(background="white")
root.geometry("1200x750+100+20")

# Create a frame
frame = Frame(root, bg="white")
frame.pack(pady=20)

# Function to add and grid images
def add_image(image_path, row, col, width=200, height=200):
    image = Image.open(image_path)
    image = image.resize((width, height))
    convert_image = ImageTk.PhotoImage(image)
    
    # Create label to hold the image
    label = tk.Label(frame, image=convert_image, bg="white")
    label.image = convert_image 
    
    # Position image using grid
    label.grid(row=row, column=col, padx=30, pady=10)


# Image paths
dog_image = r"C:\Users\ppatu\Computer-Vision-Assignments\Computer-Vision-Assignments\Assignment2\Images\dog.bmp"
bike_image = r"C:\Users\ppatu\Computer-Vision-Assignments\Computer-Vision-Assignments\Assignment2\Images\bicycle.bmp"
add_image(dog_image, 0, 0)
add_image(dog_image, 0, 1, 300, 300)
add_image(bike_image, 1, 0)
add_image(bike_image, 1, 1, 300, 300)

root.mainloop()
