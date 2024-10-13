import torch
import cv2
import os
import numpy as np
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the pre-trained Faster R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define the dataset path
dataset_path = "C:/Users/betha/OneDrive/Documents/HackDearborn24/HackDearborn24/Foggy_Driving/leftImg8bit/test/public"  # Change this to your actual path
image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg') or f.endswith('.png')]

# Define a transformation for the images (convert to tensor and normalize)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to PyTorch tensor
])

def load_image(image_path):
    """Load and preprocess an image."""
    image = Image.open(image_path).convert("RGB")  # Open the image and convert it to RGB format
    return transform(image)  # Apply the transformation (convert to tensor)

def display_image_with_boxes(image, boxes, labels):
    """Display an image with bounding boxes and labels."""
    # Convert the tensor image to numpy array and format it for Matplotlib
    image_np = image.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
    image_np = np.clip(image_np, 0, 1)  # Ensure the values are in the range [0, 1]

    # Create a Matplotlib figure
    fig, ax = plt.subplots(1)
    ax.imshow(image_np)  # Show the image

    # Add bounding boxes to the image
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box
        # Create a rectangle patch
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Add label text
        ax.text(x_min, y_min, labels[i], color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

def detect_objects_in_image(image_tensor):
    """Use the model to detect objects in the image."""
    with torch.no_grad():
        predictions = model([image_tensor])  # Perform object detection on the image
    return predictions[0]  # Return the first image's prediction

# Loop over images in the dataset
for image_file in image_files:
    print(f"Processing: {image_file}")  # Debugging line
    # Load and preprocess the image
    try:
        image_tensor = load_image(image_file)

        # Detect objects in the image
        predictions = detect_objects_in_image(image_tensor)

        # Extract the bounding boxes and labels
        boxes = predictions['boxes'].cpu().numpy()  # Convert to numpy
        labels = predictions['labels'].cpu().numpy()  # Convert to numpy
        scores = predictions['scores'].cpu().numpy()  # Confidence scores

        # Filter the detections by confidence score (optional)
        confidence_threshold = 0.5
        selected_indices = scores >= confidence_threshold
        boxes = boxes[selected_indices]
        labels = labels[selected_indices]

        # Convert label indices to class names
        COCO_INSTANCE_CATEGORY_NAMES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
            'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'TV',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
            'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        labels = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in labels]

        # Display the image with bounding boxes
        display_image_with_boxes(image_tensor, boxes, labels)

        # Wait for user input to proceed to the next image
        input("Press Enter to continue to the next image...")
    except Exception as e:
        print(f"Error processing image {image_file}: {e}")
