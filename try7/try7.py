import cv2
import numpy as np


def dark_channel(image, size=9):  # Reduced size for dark channel
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

    # Debugging: Check the transmission values
    print("Transmission min:", np.min(transmission), "max:", np.max(transmission))

    # Recover the image
    recovered_image = np.empty_like(image, dtype=np.float32)
    for i in range(3):  # For each channel
        recovered_image[:, :, i] = (
            image[:, :, i] - atmospheric_light[0, 0, i]
        ) / transmission[:, :, 0] + atmospheric_light[0, 0, i]

    recovered_image = np.clip(recovered_image, 0, 255).astype(np.uint8)
    return recovered_image


def dehaze_image(image_path, output_path):
    # Load the foggy image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found. Please check the input path.")
        return

    # Step 1: Compute the dark channel
    dark_channel_image = dark_channel(image)

    # Debugging: Display the dark channel
    cv2.imshow("Dark Channel", dark_channel_image)

    # Step 2: Estimate atmospheric light
    atmospheric_light = estimate_atmospheric_light(image, dark_channel_image)
    print("Estimated Atmospheric Light:", atmospheric_light)

    # Step 3: Recover the image
    recovered_image = recover_image(image, dark_channel_image, atmospheric_light)

    # Save the result
    cv2.imwrite(output_path, recovered_image)

    # Display the original and dehazed images
    cv2.imshow("Original Image", image)
    cv2.imshow("Dehazed Image", recovered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
dehaze_image(
    "C:/Users/hassa/OneDrive - Umich/Desktop/2024 HACKATHON/try7/foggyimage_person.jpg",
    "C:/Users/hassa/OneDrive - Umich/Desktop/2024 HACKATHON/try7/dehazed_image_person.jpg",
)
