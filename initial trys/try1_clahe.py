import cv2
import numpy as np
import os  # Import the os module


# Function to apply CLAHE in the YUV color space
def enhance_contrast_color(image):
    # Convert the image from BGR to YUV color space
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # Split the YUV channels
    y_channel, u_channel, v_channel = cv2.split(yuv)

    # Create a CLAHE object for the Y channel
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(2, 2))

    # Apply CLAHE to the Y channel (luminance)
    enhanced_y_channel = clahe.apply(y_channel)

    # Merge the enhanced Y channel back with U and V channels
    enhanced_yuv = cv2.merge((enhanced_y_channel, u_channel, v_channel))

    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(enhanced_yuv, cv2.COLOR_YUV2BGR)

    return enhanced_image


# Load the video
input_video_path = "C:/Users/hassa/OneDrive - Umich/Desktop/2024 HACKATHON/foggyvideo.mp4"  # Replace with your video path
output_video_path = "C:/Users/hassa/OneDrive - Umich/Desktop/2024 HACKATHON/enhanced_video.mp4"  # Replace with your desired output path

# Open the video file
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of the frames
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of the frames

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    while True:
        ret, frame = cap.read()  # Read a frame from the video

        if not ret:  # Break the loop if there are no frames left
            break

        # Enhance the contrast of the current frame
        enhanced_frame = enhance_contrast_color(frame)

        # Write the enhanced frame to the output video
        out.write(enhanced_frame)

        # Optional: Display the original and enhanced frames (for debugging)
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

# Open the enhanced video with VLC
vlc_path = "C:/Program Files/VideoLAN/VLC/vlc.exe"  # Adjust the path to your VLC installation if necessary
os.system(
    f'"{vlc_path}" "{output_video_path}"'
)  # Command to run VLC with the output video
