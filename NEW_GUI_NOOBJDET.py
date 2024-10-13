from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QPushButton, QFileDialog, QHBoxLayout, QFrame
import sys
import cv2
from PyQt5.QtGui import QImage, QPixmap
import numpy as np

class ObjectDetectionApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Setting up the GUI layout
        self.setStyleSheet("background-color: #f0f0f0;")
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(20)

        # Image Display Widget
        self.image_label = QLabel(self)
        self.image_label.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.image_label.setStyleSheet("background-color: #ffffff; border: 2px solid #d3d3d3;")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        # Live Camera Display Widget for Comparison
        self.live_camera_label = QLabel(self)
        self.live_camera_label.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.live_camera_label.setStyleSheet("background-color: #ffffff; border: 2px solid #d3d3d3;")
        self.live_camera_label.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.live_camera_label)

        # Horizontal Layout for Buttons
        self.button_layout = QHBoxLayout()
        self.button_layout.setSpacing(20)

        # Load Image Button
        self.load_button = QPushButton('Load Image', self)
        self.load_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; font-size: 16px; border-radius: 5px;")
        self.load_button.clicked.connect(self.load_image)
        self.button_layout.addWidget(self.load_button)

        # Detect Objects Button
        self.detect_button = QPushButton('Detect Objects', self)
        self.detect_button.setStyleSheet("background-color: #2196F3; color: white; padding: 10px; font-size: 16px; border-radius: 5px;")
        self.detect_button.clicked.connect(self.detect_objects)
        self.button_layout.addWidget(self.detect_button)

        # Fog Detection Button (Live Camera)
        self.fog_button = QPushButton('Fog Detected (Live Camera)', self)
        self.fog_button.setStyleSheet("background-color: #FF5722; color: white; padding: 10px; font-size: 16px; border-radius: 5px;")
        self.fog_button.clicked.connect(self.detect_fog)
        self.button_layout.addWidget(self.fog_button)

        # Adding Button Layout to Main Layout
        self.layout.addLayout(self.button_layout)

        # Setting up the final layout
        self.setLayout(self.layout)
        self.setWindowTitle('Object Detection App')
        self.setGeometry(200, 200, 1200, 800)
        self.setWindowIcon(QtGui.QIcon('icon.png'))

    def load_image(self):
        # Function to load an image from the filesystem
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)", options=options)
        if file_name:
            self.image = cv2.imread(file_name)
            self.display_image(self.image, self.image_label)

    def display_image(self, img, label):
        # Convert OpenCV image to QImage and display it
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q_img))
        label.setFixedSize(width, height)

    def detect_objects(self):
        # This function should call the detection model and display the result
        if hasattr(self, 'image'):
            # Here we will just create a dummy detection for illustration purposes
            img_copy = self.image.copy()
            height, width, _ = img_copy.shape
            cv2.rectangle(img_copy, (width//4, height//4), (3*width//4, 3*height//4), (255, 0, 0), 4)
            cv2.putText(img_copy, 'Detected Object', (width//4, height//4 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            self.display_image(img_copy, self.image_label)

    def detect_fog(self):
        # Function to detect fog using the live camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Dehaze the frame
            dehazed_frame = self.dehaze_frame(frame)

            # Display the original and dehazed frames side by side
            self.display_image(frame, self.live_camera_label)
            self.display_image(dehazed_frame, self.image_label)

            # Break the loop when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def dark_channel(self, im, size=10):
        """Computes the dark channel of the image."""
        min_channel = cv2.erode(im, np.ones((size, size)))
        dark_channel = np.min(min_channel, axis=2)
        return dark_channel

    def estimate_atmospheric_light(self, im, dark_channel):
        """Estimates atmospheric light from the dark channel."""
        num_pixels = dark_channel.size
        num_brightest = int(max(num_pixels / 1000, 1))  # Avoid zero division
        dark_vector = dark_channel.reshape(num_pixels)
        indices = np.argsort(dark_vector)[-num_brightest:]  # Get indices of the brightest
        atmospheric_light = np.mean(im.reshape(num_pixels, 3)[indices], axis=0)
        return atmospheric_light

    def transmission_estimation(self, im, atmospheric_light, omega=0.8):
        """Estimates the transmission map."""
        normalized_image = im.astype(np.float64) / atmospheric_light
        transmission = 1 - omega * self.dark_channel(normalized_image)
        return transmission

    def recover_image(self, im, transmission, atmospheric_light, t0=0.1):
        """Recovers the haze-free image."""
        transmission = np.clip(transmission, t0, 1)
        recovered_image = (im.astype(np.float64) - atmospheric_light) / transmission[:, :, np.newaxis] + atmospheric_light
        recovered_image = np.clip(recovered_image, 0, 255).astype(np.uint8)  # Ensure valid pixel range
        return recovered_image

    def dehaze_frame(self, frame):
        """Dehazes a single frame."""
        dark_channel_map = self.dark_channel(frame)
        atmospheric_light = self.estimate_atmospheric_light(frame, dark_channel_map)
        transmission_map = self.transmission_estimation(frame, atmospheric_light)
        recovered_image = self.recover_image(frame, transmission_map, atmospheric_light)
        return recovered_image

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    mainWin = ObjectDetectionApp()
    mainWin.show()
    sys.exit(app.exec_())