import cv2
import numpy as np

# Optical Flow parameters
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Global variables for smoothing
prev_transformation = None  # To store the last transformation matrix
smooth_factor = 0.1  # Smoothing factor for the transformation

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
    transmission = 1 - t0 * dark_channel[:, :, np.newaxis] / atmospheric_light
    transmission = np.clip(transmission, 0.1, 1)
    recovered_image = np.empty_like(image, dtype=np.float32)
    
    for i in range(3):  # For each channel
        recovered_image[:, :, i] = (image[:, :, i] - atmospheric_light[0, 0, i]) / transmission[:, :, 0] + atmospheric_light[0, 0, i]

    return np.clip(recovered_image, 0, 255).astype(np.uint8)

def clahe(image):
    """Apply CLAHE to the image."""
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_l_channel = clahe.apply(l_channel)
    lab_image = cv2.merge((clahe_l_channel, a_channel, b_channel))
    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

def sharpen_image(image):
    """Sharpen the image using a kernel."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def edge_detection(image):
    """Detect edges using Canny edge detection."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray_image, 100, 200)

def stabilize_frame(frame, prev_frame, prev_points):
    """Stabilize the current frame based on the previous frame."""
    global prev_transformation
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_points is None:
        prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    curr_points, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None, **lk_params)

    if curr_points is not None and prev_points is not None:
        good_new = curr_points[st.flatten() == 1]
        good_old = prev_points[st.flatten() == 1]

        if len(good_new) >= 3:
            matrix, _ = cv2.estimateAffinePartial2D(good_old, good_new)
            if matrix is not None:
                # Extract the rotation angle from the transformation matrix
                angle = np.arctan2(matrix[1, 0], matrix[0, 0]) * (180 / np.pi)

                # Calculate the translation vector
                translation = np.array([matrix[0, 2], matrix[1, 2]])

                # Cap the maximum displacement
                max_displacement = 10
                translation = np.clip(translation, -max_displacement, max_displacement)

                # Create the transformation matrix
                # Negate the angle for the inverse rotation
                rotation_matrix = cv2.getRotationMatrix2D((frame.shape[1] // 2, frame.shape[0] // 2), -angle, 1)
                rotation_matrix[0, 2] += translation[0]
                rotation_matrix[1, 2] += translation[1]

                # If there's a previous transformation, combine it
                if prev_transformation is not None:
                    rotation_matrix = (1 - smooth_factor) * prev_transformation + smooth_factor * rotation_matrix

                # Update the previous transformation
                prev_transformation = rotation_matrix

                # Apply the transformation
                stabilized_frame = cv2.warpAffine(frame, rotation_matrix, (frame.shape[1], frame.shape[0]))

                return stabilized_frame, good_new

    return frame, prev_points

def process_webcam():
    """Process webcam feed for stabilization and enhancement."""
    cap = cv2.VideoCapture(1)

    prev_frame = None
    prev_points = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Stabilize the frame
        if prev_frame is not None:
            stabilized_frame, prev_points = stabilize_frame(frame, prev_frame, prev_points)
        else:
            stabilized_frame = frame

        # Apply image processing techniques
        enhanced_image = clahe(stabilized_frame)
        sharpened_image = sharpen_image(enhanced_image)
        edges = edge_detection(sharpened_image)

        # Display images in separate windows
        cv2.imshow("Original Image", stabilized_frame)
        cv2.imshow("Enhanced Image", sharpened_image)
        cv2.imshow("Edge Detection", edges)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_frame = frame

    cap.release()
    cv2.destroyAllWindows()

# Start webcam processing
process_webcam()
