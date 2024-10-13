import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the webcam (use '0' for the default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Process the webcam feed frame-by-frame
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the webcam

    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLOv8 detection on the frame with a confidence threshold of 0.4
    results = model(frame, conf=0.55)

    # Get the annotated frame from YOLOv8
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("YOLOv8 Webcam", annotated_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close the display window
cap.release()
cv2.destroyAllWindows()
