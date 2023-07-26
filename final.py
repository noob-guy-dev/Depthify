import customtkinter as ctk
import tkinter
from PIL import Image, ImageTk
from tkinter import filedialog
import sys
#MiDaS code

import cv2
import torch



opened_image_path = ""
depth_photo = None

def generate():
    global opened_image_path, depth_photo

    if opened_image_path:
        model_type = style_dropdown.get()

        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas.to(device)
        midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type in ["DPT_Large", "DPT_Hybrid"]:
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform

        img = cv2.imread(opened_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = transform(img).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        output = prediction.cpu().numpy()
        output = (output / output.max() * 255).astype('uint8')  # Convert to uint8
        depth_img = Image.fromarray(output)  # Convert to PIL.Image

        # Calculate the aspect ratio
        original_width, original_height = img.shape[1], img.shape[0]
        max_size = 512
        ratio = min(max_size / original_width, max_size / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)

        # Resize the depth map image to match the canvas size while preserving aspect ratio
        depth_img = depth_img.resize((new_width, new_height), Image.LANCZOS)

        # Create a PhotoImage for the resized depth map
        depth_photo = ImageTk.PhotoImage(depth_img)

        # Adjust canvas size to match image
        canvas.config(width=new_width, height=new_height)
        canvas.create_image(0, 0, anchor=tkinter.NW, image=depth_photo)
        canvas.image = depth_photo
    else:
        print("No image opened yet!")



def open_image():
    global opened_image_path
    if file_path := filedialog.askopenfilename(
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")]
    ):
        img = Image.open(file_path)

        # Calculate new width and height to maintain the aspect ratio
        original_width, original_height = img.size
        max_size = 512  # Set the maximum size for the width or height
        ratio = min(max_size / original_width, max_size / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)

        # Resize the image
        img = img.resize((new_width, new_height), Image.LANCZOS)

        photo = ImageTk.PhotoImage(img)

        # Adjust canvas size to match image
        canvas.config(width=new_width, height=new_height)
        canvas.create_image(0, 0, anchor=tkinter.NW, image=photo)
        canvas.image = photo  # Keep a reference to avoid garbage collection issues

        # Enable the canvas
        canvas.config(state=tkinter.NORMAL)
        opened_image_path = file_path

def save_depthmap():
    global depth_photo
    if depth_photo:
        if save_path := filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG files", "*.png")]
        ):
            # Convert depth_photo to PIL Image
            depth_img = ImageTk.getimage(depth_photo)
            depth_img.save(save_path)
            print("Depth map saved successfully.")
    else:
        print("No depth map to save.")


def show_depthmap():
    global opened_image_path, depth_photo
    
    if opened_image_path and depth_photo:
        # Create a new window to show the depth map side by side with the original image
        depthmap_window = tkinter.Toplevel(root)
        depthmap_window.title("Depth Map")
        
        # Create a frame to hold the canvas widgets
        frame = ctk.CTkFrame(depthmap_window)
        frame.pack(expand=True, padx=20, pady=20)

        # Load the original image
        img = Image.open(opened_image_path)
        
        # Calculate the aspect ratio to adjust the original image size to match the depth map
        original_width, original_height = img.size
        depth_width, depth_height = depth_photo.width(), depth_photo.height()
        ratio = min(depth_width / original_width, depth_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        
        # Resize the original image
        img = img.resize((new_width, new_height), Image.LANCZOS)

        original_photo = ImageTk.PhotoImage(img)
        original_canvas = tkinter.Canvas(frame, bg="#1C1D1C", width=new_width, height=new_height)
        original_canvas.pack(side="left")
        original_canvas.create_image(0, 0, anchor=tkinter.NW, image=original_photo)
        original_canvas.image = original_photo
        
        # Create a label to separate the two canvas widgets
        #ctk.CTkLabel(frame, text="Depth Map").pack(side="left", padx=10, pady=10)
        
        # Display the generated depth map
        depth_canvas = tkinter.Canvas(frame, bg="#1C1D1C", width=depth_width, height=depth_height)
        depth_canvas.pack(side="left")
        depth_canvas.create_image(0, 0, anchor=tkinter.NW, image=depth_photo)
        depth_canvas.image = depth_photo
    else:
        print("No image opened yet or depth map not generated.")


def destroy():
    sys.exit()


root = ctk.CTk()
root.title("Depthify")

ctk.set_appearance_mode("dark")

input_frame = ctk.CTkFrame(root)
input_frame.pack(side="left", expand=True, padx=20, pady=20)

style_label = ctk.CTkLabel(input_frame, text="Style")
style_label.grid(row=1, column=0, padx=10, pady=10)
style_dropdown = ctk.CTkComboBox(input_frame, values=["MiDaS_small", "DPT_Hybrid", "DPT_Large"])
style_dropdown.grid(row=1, column=1, padx=10, pady=10)



generate_button = ctk.CTkButton(input_frame, text="Generate", command=generate)
generate_button.grid(row=4, column=0, columnspan=2, sticky="news", padx=10, pady=10)

# Adjust canvas size to match image
canvas = tkinter.Canvas(root, bg="#1C1D1C", width=400, height=700)
canvas.pack(side="left")

# Disable the canvas initially
canvas.config(state=tkinter.DISABLED)

# Create a button to open the image file
open_button = ctk.CTkButton(input_frame, text="Open Image", command=open_image)
open_button.grid(row=3, column=0, columnspan=2, sticky="news", padx=10, pady=10)

save_button = ctk.CTkButton(input_frame, text="Save Depthmap", command=save_depthmap)
save_button.grid(row=5, column=0, columnspan=2, sticky="news", padx=10, pady=10)

show_button = ctk.CTkButton(input_frame, text="Show Depthmap", command=show_depthmap)
show_button.grid(row=6, column=0, columnspan=2, sticky="news", padx=10, pady=10)

show_button = ctk.CTkButton(input_frame, text="Exit", command=destroy)
show_button.grid(row=7, column=0, columnspan=2, sticky="news", padx=10, pady=10)

root.mainloop()
