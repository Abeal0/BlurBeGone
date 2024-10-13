import cv2
import numpy as np


def reduce_fogginess_and_blurriness(image_path, output_path):
    # Load the foggy image
    image = cv2.imread(image_path)

    # Check if the image was loaded
    if image is None:
        print("Error loading image. Please check the file path.")
        return

    # Apply CLAHE for dehazing
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(2, 2))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    image_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Deblurring using a simple method (unsharp mask)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image_deblurred = cv2.filter2D(image_clahe, -1, kernel)

    # Save the result
    cv2.imwrite(output_path, image_deblurred)


# Example usage
reduce_fogginess_and_blurriness(
    "C:/Users/hassa/OneDrive - Umich/Desktop/2024 HACKATHON/try6/foggyperson.jpg",
    "C:/Users/hassa/OneDrive - Umich/Desktop/2024 HACKATHON/try6/clear_image_person.jpg",
)
