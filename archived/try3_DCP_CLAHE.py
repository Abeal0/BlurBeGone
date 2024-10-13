import cv2
import numpy as np


def dark_channel(im, size=15):
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


def transmission_estimation(im, atmospheric_light, omega=0.95):
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
    recovered_image = np.clip(recovered_image, 0, 255).astype(
        np.uint8
    )  # Ensure valid pixel range
    return recovered_image


def enhance_contrast_color(image):
    """Applies CLAHE to enhance contrast in the YUV color space."""
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y_channel, u_channel, v_channel = cv2.split(yuv)

    clahe = cv2.createCLAHE(
        clipLimit=10.0, tileGridSize=(8, 8)
    )  # Adjust tileGridSize as needed
    enhanced_y_channel = clahe.apply(y_channel)

    enhanced_yuv = cv2.merge((enhanced_y_channel, u_channel, v_channel))
    enhanced_image = cv2.cvtColor(enhanced_yuv, cv2.COLOR_YUV2BGR)

    return enhanced_image


def process_video(input_video_path, output_video_path):
    """Main function to process the video for dehazing and contrast enhancement."""
    print("Opening video...")
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print("Processing frames...")
    while True:
        ret, frame = cap.read()  # Read a frame from the video

        if not ret:
            print("No frame returned, breaking loop...")
            break

        # Step 1: Dehaze the current frame
        try:
            dark_channel_map = dark_channel(frame)
            atmospheric_light = estimate_atmospheric_light(frame, dark_channel_map)
            transmission_map = transmission_estimation(frame, atmospheric_light)
            dehazed_frame = recover_image(frame, transmission_map, atmospheric_light)
        except Exception as e:
            print(f"Error during dehazing: {e}")
            continue  # Skip this frame if there's an error

        # Step 2: Enhance the contrast of the dehazed frame using CLAHE
        try:
            enhanced_frame = enhance_contrast_color(dehazed_frame)
        except Exception as e:
            print(f"Error during contrast enhancement: {e}")
            continue  # Skip this frame if there's an error

        # Write the enhanced frame to the output video
        out.write(enhanced_frame)

        # Optional: Display the original and enhanced frames
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Enhanced Frame", enhanced_frame)

        # Press 'q' to exit early
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Processing complete. The enhanced video is saved as:", output_video_path)


# Example usage
input_video_path = r"C:/Users/hassa/OneDrive - Umich\Desktop/2024 HACKATHON/foggyvideo.mp4"  # Replace with your video path
output_video_path = r"C:/Users/hassa/OneDrive - Umich\Desktop/2024 HACKATHON/enhanced_video.mp4"  # Replace with your desired output path

process_video(input_video_path, output_video_path)
