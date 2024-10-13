import tkinter as tk
from tkinter import filedialog, Label
import cv2
from PIL import Image, ImageTk
import numpy as np
from ultralytics import YOLO


# Function to apply CLAHE and increase contrast in an image
def apply_clahe(image):
    """Applies CLAHE to enhance contrast in the image."""
    # Convert image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split into L, A, B channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L (lightness) channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L channel back with A and B channels
    enhanced_lab = cv2.merge((cl, a, b))
    
    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_image


# Dark Channel Prior functions for dehazing
def dark_channel(im, size=10):
    """Computes the dark channel of the image."""
    min_channel = cv2.erode(im, np.ones((size, size)))
    dark_channel = np.min(min_channel, axis=2)
    return dark_channel


def estimate_atmospheric_light(im, dark_channel):
    """Estimates atmospheric light from the dark channel."""
    num_pixels = dark_channel.size
    num_brightest = int(max(num_pixels / 1000, 1))  # Avoid zero division
    dark_vector = dark_channel.reshape(num_pixels)
    indices = np.argsort(dark_vector)[-num_brightest:]  # Get indices of the brightest
    atmospheric_light = np.mean(im.reshape(num_pixels, 3)[indices], axis=0)
    return atmospheric_light


def transmission_estimation(im, atmospheric_light, omega=0.8):
    """Estimates the transmission map."""
    normalized_image = im.astype(np.float64) / atmospheric_light
    transmission = 1 - omega * dark_channel(normalized_image)
    return transmission


def recover_image(im, transmission, atmospheric_light, t0=0.1):
    """Recovers the haze-free image."""
    transmission = np.clip(transmission, t0, 1)
    recovered_image = (im.astype(np.float64) - atmospheric_light) / transmission[
        :, :, np.newaxis
    ] + atmospheric_light
    recovered_image = np.clip(recovered_image, 0, 255).astype(np.uint8)
    return recovered_image


# Function to load an image file
def load_image():
    global img, photo
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        display_image(img)


# Function to display the image in the GUI
def display_image(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    photo = ImageTk.PhotoImage(image=pil_image)
    image_label.config(image=photo)
    image_label.image = photo


# Object detection and road line highlighting using YOLOv8
def detect_objects(image):
    model = YOLO('yolov8n.pt')  # Load YOLOv8 model

    # Run the object detection
    results = model(image)

    # Process detection results
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        label = box.cls[0]
        
        # Label specific classes
        label_name = model.names[int(label)]
        if label_name in ['person', 'road', 'car', 'bus', 'truck', 'bicycle']:  # Add more relevant classes as needed
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{label_name} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)

        # Highlight road lines using edge detection if road is detected
        if label_name == 'road':
            road_region = image[y1:y2, x1:x2]
            road_with_lines = highlight_road_lines(road_region)
            image[y1:y2, x1:x2] = road_with_lines

    return image


# Function to highlight road lines using Canny edge detection and Hough Line Transform
def highlight_road_lines(road_region):
    gray_road = cv2.cvtColor(road_region, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray_road, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(road_region, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return road_region


# Function to perform dehazing and CLAHE contrast enhancement
def dehaze_and_enhance(image_path):
    im = cv2.imread(image_path)
    if im is None:
        print("Error: Image not found!")
        return

    dark_channel_map = dark_channel(im)
    atmospheric_light = estimate_atmospheric_light(im, dark_channel_map)
    transmission_map = transmission_estimation(im, atmospheric_light)
    recovered_image = recover_image(im, transmission_map, atmospheric_light)

    # Apply CLAHE for contrast enhancement
    enhanced_image = apply_clahe(recovered_image)

    # Detect objects and highlight road lines
    final_image = detect_objects(enhanced_image)

    display_image(final_image)


# Main GUI window
window = tk.Tk()
window.title("Object Detection, Dehazing, and CLAHE Enhancement")
window.geometry("800x600")

# Button to load image
load_btn = tk.Button(window, text="Load and Process Image", command=lambda: dehaze_and_enhance(filedialog.askopenfilename()))
load_btn.pack()

# Label to display image
image_label = Label(window)
image_label.pack()

# Start the Tkinter main loop
window.mainloop()
