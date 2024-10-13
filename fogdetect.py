import cv2
import numpy as np

def compute_laplacian_variance(image):
    """Calculate the variance of the Laplacian to measure sharpness."""
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    variance = laplacian.var()  # Compute the variance of the Laplacian
    return variance

def is_foggy(variance, threshold=100):
    """Determine if an image is foggy based on variance of Laplacian."""
    return variance < threshold  # Return True if the image is considered foggy

def fog_detection(image, laplacian_threshold=100):
    """Main fog detection function based on sharpness."""
    laplacian_variance = compute_laplacian_variance(image)
    return is_foggy(laplacian_variance, laplacian_threshold)

# Example usage
if __name__ == "__main__":
    cap = cv2.VideoCapture(1)  # Use 0 for the primary camera, 1 for the secondary

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Detect fog based on sharpness
        if fog_detection(frame):
            cv2.putText(frame, "Fog Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Fog Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Fog Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
