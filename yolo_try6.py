import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QVBoxLayout,
    QWidget,
    QGridLayout,
    QPushButton,
    QSizePolicy,
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt
from ultralytics import YOLO


# Fog detection functions
def compute_laplacian_variance(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    variance = laplacian.var()
    return variance


def is_foggy(variance, threshold=100):
    return variance < threshold


def fog_detection(image, laplacian_threshold=100):
    laplacian_variance = compute_laplacian_variance(image)
    return is_foggy(laplacian_variance, laplacian_threshold)


def dark_channel(image, size=15):
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel


def estimate_atmospheric_light(image, dark_channel):
    h, w = dark_channel.shape
    num_pixels = h * w
    num_brightest = int(max(num_pixels // 1000, 1))
    indices = np.argpartition(dark_channel.flatten(), -num_brightest)[-num_brightest:]
    return np.mean(image.reshape(-1, 3)[indices], axis=0)


def recover_image(image, dark_channel, atmospheric_light, t0=0.1):
    atmospheric_light = atmospheric_light.reshape(1, 1, 3)
    transmission = 1 - t0 * dark_channel[:, :, np.newaxis] / atmospheric_light
    transmission = np.clip(transmission, 0.1, 1)

    recovered_image = np.empty_like(image, dtype=np.float32)
    for i in range(3):
        recovered_image[:, :, i] = (
            image[:, :, i] - atmospheric_light[0, 0, i]
        ) / transmission[:, :, 0] + atmospheric_light[0, 0, i]

    return np.clip(recovered_image, 0, 255).astype(np.uint8)


def clahe(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_l_channel = clahe.apply(l_channel)
    lab_image = cv2.merge((clahe_l_channel, a_channel, b_channel))
    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)


def edge_detection(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray_image, 100, 200)


class ImageDisplay(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Processing GUI")
        self.setFixedSize(1024, 768)  # Set a fixed window size

        # Create layout
        self.grid_layout = QGridLayout(self)

        # Add logo label
        self.logo_label = QLabel()
        logo_pixmap = QPixmap(
            "C:/Users/hassa/Downloads/bluebegone.jpg"
        )  # Update the path if needed
        self.logo_label.setPixmap(
            logo_pixmap.scaled(2400, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.grid_layout.addWidget(
            self.logo_label, 0, 0, 1, 2
        )  # Span both columns to center

        # Add a button to start processing
        self.start_button = QPushButton("Start Processing")
        self.start_button.setStyleSheet(
            "background-color: lightblue; font-size: 24px; padding: 15px;"
        )
        self.start_button.clicked.connect(self.start_processing)
        self.grid_layout.addWidget(
            self.start_button, 1, 0, 1, 2
        )  # Span both columns to center

        # Add fog detected alert (initially hidden)
        self.fog_alert_label = QLabel("Fog Detected!")
        self.fog_alert_label.setStyleSheet(
            "background-color: lightblue; font-size: 24px; padding: 10px; margin-top: 10px;"
        )
        self.fog_alert_label.setAlignment(Qt.AlignCenter)
        self.fog_alert_label.hide()  # Initially hidden
        self.grid_layout.addWidget(
            self.fog_alert_label, 0, 0, 1, 2
        )  # Span both columns

        # Image labels (not visible initially)
        self.original_label = None
        self.enhanced_label = None
        self.edges_label = None
        self.annotated_label = None  # Label for annotated frame

    def start_processing(self):
        """Start webcam processing, hide logo and button, and show fog alert when needed."""
        # Hide the logo and start button when processing starts
        self.logo_label.hide()
        self.start_button.hide()

        # Initialize image labels dynamically with consistent sizes
        self.original_label = QLabel("Original Image")
        self.enhanced_label = QLabel("Enhanced Image")
        self.edges_label = QLabel("Edge Detection")
        self.annotated_label = QLabel("YOLOv8 Detection")  # Label for YOLO detection

        # Apply consistent font and alignment settings
        for label in [
            self.original_label,
            self.enhanced_label,
            self.edges_label,
            self.annotated_label,
        ]:
            label.setFont(QFont("Arial", 16))
            label.setAlignment(Qt.AlignCenter)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            # Set fixed size for the labels to keep them consistent
            label.setFixedSize(480, 360)  # Example size (you can adjust)

        # Organize the labels into the grid layout
        self.grid_layout.addWidget(self.original_label, 1, 0)  # Top left
        self.grid_layout.addWidget(self.enhanced_label, 1, 1)  # Top right
        self.grid_layout.addWidget(self.edges_label, 2, 0)  # Bottom left
        self.grid_layout.addWidget(self.annotated_label, 2, 1)  # Bottom right

        self.process_webcam()

    def process_webcam(self):
        cap = cv2.VideoCapture(0)

        # Load the YOLOv8 model once before the loop
        model = YOLO("yolov8n.pt")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            # Check for fog detection
            if fog_detection(frame):
                dark_channel_image = dark_channel(frame)
                atmospheric_light = estimate_atmospheric_light(
                    frame, dark_channel_image
                )
                recovered_image = recover_image(
                    frame, dark_channel_image, atmospheric_light
                )
                enhanced_image = clahe(recovered_image)
                edges = edge_detection(enhanced_image)

                # Show fog alert
                self.fog_alert_label.show()

                # Run YOLOv8 detection on the enhanced image
                results = model(enhanced_image, conf=0.55)
                annotated_frame = results[0].plot()

                # Update GUI with processed images
                self.display_image(frame, self.original_label)  # Original frame
                self.display_image(
                    enhanced_image, self.enhanced_label
                )  # Enhanced image
                self.display_image(edges, self.edges_label)  # Edge detection image
                self.display_image(
                    annotated_frame, self.annotated_label
                )  # YOLO detection image
            else:
                # Hide fog alert
                self.fog_alert_label.hide()

                # Only show the original image
                self.display_image(frame, self.original_label)
                self.enhanced_label.clear()
                self.edges_label.clear()
                self.annotated_label.clear()  # Clear YOLO detection label

            self.resize_labels()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def display_image(self, img, label):
        """Convert an OpenCV image to QImage and display it in a QLabel."""
        if len(img.shape) == 3:
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            qimg = QImage(img.data, width, height, bytes_per_line, QImage.Format_BGR888)
        else:
            height, width = img.shape
            qimg = QImage(img.data, width, height, width, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(
            pixmap.scaled(
                label.width(),
                label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def resize_labels(self):
        """Adjust the size of the labels to ensure they all maintain consistent dimensions."""
        for label in [
            self.original_label,
            self.enhanced_label,
            self.edges_label,
            self.annotated_label,
        ]:
            label.setFixedSize(480, 360)  # Ensure consistent label sizes


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageDisplay()
    window.show()
    sys.exit(app.exec_())
