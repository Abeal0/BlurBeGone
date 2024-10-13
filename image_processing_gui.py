import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QGridLayout, QPushButton, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt

def dark_channel(image, size=15):
    """Calculate the dark channel of an image."""
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

def estimate_atmospheric_light(image, dark_channel):
    """Estimate atmospheric light in the image."""
    h, w = dark_channel.shape
    num_pixels = h * w
    num_brightest = int(max(num_pixels // 1000, 1))  # Top 0.1% brightest pixels
    indices = np.argpartition(dark_channel.flatten(), -num_brightest)[-num_brightest:]
    return np.mean(image.reshape(-1, 3)[indices], axis=0)

def recover_image(image, dark_channel, atmospheric_light, t0=0.1):
    """Recover the scene radiance."""
    atmospheric_light = atmospheric_light.reshape(1, 1, 3)  # Reshape for broadcasting

    # Compute the transmission map
    transmission = 1 - t0 * dark_channel[:, :, np.newaxis] / atmospheric_light
    transmission = np.clip(transmission, 0.1, 1)

    # Recover the image
    recovered_image = np.empty_like(image, dtype=np.float32)
    for i in range(3):  # For each channel
        recovered_image[:, :, i] = (
            image[:, :, i] - atmospheric_light[0, 0, i]
        ) / transmission[:, :, 0] + atmospheric_light[0, 0, i]

    recovered_image = np.clip(recovered_image, 0, 255).astype(np.uint8)
    return recovered_image

def clahe(image):
    """Apply CLAHE to the image."""
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_l_channel = clahe.apply(l_channel)

    # Merge channels back
    lab_image = cv2.merge((clahe_l_channel, a_channel, b_channel))
    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

def sharpen_image(image):
    """Sharpen the image using a kernel."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

def edge_detection(image):
    """Detect edges using Canny edge detection."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    return edges

class ImageDisplay(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Processing GUI")
        self.setGeometry(100, 100, 1024, 768)  # Increased size

        # Create layout
        self.grid_layout = QGridLayout(self)

        # Header
        header_label = QLabel("Blur Be Gone")
        header_label.setFont(QFont("Arial", 32))  # Increased font size for header
        header_label.setStyleSheet("font-weight: bold; color: darkblue; text-align: center;")
        header_label.setAlignment(Qt.AlignCenter)  # Center the header
        self.grid_layout.addWidget(header_label, 0, 0, 1, 2)

        # Image labels
        self.original_label = QLabel("Original Image")
        self.enhanced_label = QLabel("Enhanced Image")
        self.edges_label = QLabel("Edge Detection")

        # Set font sizes and policies
        for label in [self.original_label, self.enhanced_label, self.edges_label]:
            label.setFont(QFont("Arial", 24))  # Increased font size for labels
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            label.setAlignment(Qt.AlignCenter)  # Center the labels

        self.grid_layout.addWidget(self.original_label, 1, 0)
        self.grid_layout.addWidget(self.enhanced_label, 1, 1)
        self.grid_layout.addWidget(self.edges_label, 2, 0, 1, 2)

        # Add a button to start processing
        self.start_button = QPushButton("Fog Detected")
        self.start_button.setStyleSheet("background-color: lightblue; font-size: 24px; padding: 15px;")  # Button color and size
        self.start_button.clicked.connect(self.process_webcam)
        self.grid_layout.addWidget(self.start_button, 3, 0, 1, 2)

    def process_webcam(self):
        cap = cv2.VideoCapture(1)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            # Step 1: Compute the dark channel
            dark_channel_image = dark_channel(frame)

            # Step 2: Estimate atmospheric light
            atmospheric_light = estimate_atmospheric_light(frame, dark_channel_image)

            # Step 3: Recover the image
            recovered_image = recover_image(frame, dark_channel_image, atmospheric_light)

            # Step 4: Apply CLAHE to the recovered image
            enhanced_image = clahe(recovered_image)

            # Step 5: Detect edges
            edges = edge_detection(enhanced_image)

            # Convert images to QImage format and display them
            self.display_image(frame, self.original_label)
            self.display_image(enhanced_image, self.enhanced_label)
            self.display_image(edges, self.edges_label)

            # Update the layout and labels to resize dynamically
            self.resize_labels()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def display_image(self, img, label):
        """Convert an OpenCV image to QImage and display it in a QLabel."""
        # Check if the image is grayscale or color
        if len(img.shape) == 3:  # Color image
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        elif len(img.shape) == 2:  # Grayscale image
            height, width = img.shape
            bytes_per_line = width  # For grayscale, each pixel is one byte
            q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            raise ValueError("Invalid image shape")

        # Display the image while maintaining the aspect ratio
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)

    def resize_labels(self):
        """Resize labels dynamically based on window size."""
        for label in [self.original_label, self.enhanced_label, self.edges_label]:
            label.setFixedHeight(self.height() // 3)  # Adjust height based on window size

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageDisplay()
    window.show()
    sys.exit(app.exec_())
