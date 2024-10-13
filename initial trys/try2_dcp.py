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
    # Get the top 0.1% of the dark channel to find atmospheric light
    num_pixels = dark_channel.size
    num_brightest = int(max(num_pixels / 1000, 1))  # Avoid zero division
    dark_vector = dark_channel.reshape(num_pixels)
    indices = np.argsort(dark_vector)[-num_brightest:]  # Get indices of the brightest
    atmospheric_light = np.mean(im.reshape(num_pixels, 3)[indices], axis=0)
    return atmospheric_light


def transmission_estimation(im, atmospheric_light, omega=0.8):
    """Estimates the transmission map."""
    # Normalize the image and calculate transmission
    normalized_image = im.astype(np.float64) / atmospheric_light
    transmission = 1 - omega * dark_channel(normalized_image)
    return transmission


def recover_image(im, transmission, atmospheric_light, t0=0.1):
    """Recovers the haze-free image."""
    # Ensure the transmission is greater than t0 to avoid division by zero
    transmission = np.clip(transmission, t0, 1)
    recovered_image = (im.astype(np.float64) - atmospheric_light) / transmission[
        :, :, np.newaxis
    ] + atmospheric_light
    recovered_image = np.clip(recovered_image, 0, 255).astype(
        np.uint8
    )  # Ensure valid pixel range
    return recovered_image


def dehaze_image(image_path, output_path):
    """Main function to dehaze an image."""
    # Load the image
    im = cv2.imread(image_path)
    if im is None:
        print("Error: Image not found!")
        return

    # Step 1: Compute the dark channel
    dark_channel_map = dark_channel(im)

    # Step 2: Estimate atmospheric light
    atmospheric_light = estimate_atmospheric_light(im, dark_channel_map)

    # Step 3: Estimate the transmission map
    transmission_map = transmission_estimation(im, atmospheric_light)

    # Step 4: Recover the scene radiance
    recovered_image = recover_image(im, transmission_map, atmospheric_light)

    # Save the output image
    cv2.imwrite(output_path, recovered_image)
    print(f"Dehazed image saved as: {output_path}")

    # Display the results
    cv2.imshow("Original Image", im)
    cv2.imshow("Dehazed Image", recovered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
input_image_path = "C:/Users/hassa/OneDrive - Umich/Desktop/2024 HACKATHON/foggyimage.png"  # Replace with your image path
output_image_path = "C:/Users/hassa/OneDrive - Umich/Desktop/2024 HACKATHON/dehazed_image.png"  # Replace with desired output path

dehaze_image(input_image_path, output_image_path)
