import cv2
import numpy as np

def dark_channel(im, size=10):
    """Computes the dark channel of the image."""
    # Get the minimum value in a local patch for each channel
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
    # Step 1: Compute the dark channel
    dark_channel_map = dark_channel(frame)

    # Step 2: Estimate atmospheric light
    atmospheric_light = estimate_atmospheric_light(frame, dark_channel_map)

    # Step 3: Estimate the transmission map
    transmission_map = transmission_estimation(frame, atmospheric_light)

    # Step 4: Recover the scene radiance
    recovered_image = recover_image(frame, transmission_map, atmospheric_light)

    return recovered_image

def main():
    """Main function to capture video from the webcam and dehaze frames."""
    # Open a connection to the webcam (0 is the default camera)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Dehaze the current frame
        dehazed_frame = dehaze_frame(frame)

        # Display the original and dehazed frames
        cv2.imshow("Original Image", frame)
        cv2.imshow("Dehazed Image", dehazed_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Run the main function
if __name__ == "__main__":
    main()
