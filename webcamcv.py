import cv2
import numpy as np

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
    recovered_image = (im.astype(np.float64) - atmospheric_light) / transmission[:, :, np.newaxis] + atmospheric_light
    recovered_image = np.clip(recovered_image, 0, 255).astype(np.uint8)  # Ensure valid pixel range
    return recovered_image

def dehaze_frame(frame):
    """Dehazes a single frame."""
    dark_channel_map = dark_channel(frame)
    atmospheric_light = estimate_atmospheric_light(frame, dark_channel_map)
    transmission_map = transmission_estimation(frame, atmospheric_light)
    recovered_image = recover_image(frame, transmission_map, atmospheric_light)
    return recovered_image

def outline_image(image):
    """Applies Canny edge detection to outline the image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    outlined_image = cv2.addWeighted(image, 0.7, edges_colored, 0.3, 0)
    return outlined_image

def hough_line_detection(image):
    """Applies Hough Line Transform to detect lines in the image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw the detected lines in green

    return image

def main():
    """Main function to capture video from the webcam and dehaze frames."""
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        dehazed_frame = dehaze_frame(frame)
        outlined_frame = outline_image(frame)
        outlined_dehazed_frame = outline_image(dehazed_frame)

        # Apply Hough Line Detection to the outlined frames
        hough_frame = hough_line_detection(outlined_frame.copy())
        hough_dehazed_frame = hough_line_detection(outlined_dehazed_frame.copy())

        # Display the original, outlined, and dehazed images with Hough lines
        cv2.imshow("Original Image with Hough Lines", hough_frame)
        cv2.imshow("Dehazed Image with Hough Lines", hough_dehazed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the main function
if __name__ == "__main__":
    main()
