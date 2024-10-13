# HackDearborn24

# How to run the GUI:
To run the GUI that ouputs the processed images, you must run the yolo_final.py file. You may need to adjust  cap = cv2.VideoCapture(0) and change the 0 to a 1 depending on the enviorment you are running the code. Additonally:

logo_pixmap = QPixmap(
            "C:/Users/hassa/Downloads/bluebegone.jpg"

You will need to update the file to a photo bath on your computer. Feel free to download the "logo.png" file and utilize this!

## Inspiration
Foggy weather significantly affects driver experience and autonomous vehicle modelling, limiting visibility and safety while driving.

Studies have even found that more than 16,000 people get injured in accidents caused by fog each year, while more than 600 pass away. This is due to the drastically reduced visibility caused by fog and adverse weather conditions (Julian Sanders J.D).

That's why we wanted to create a low-cost solution that can be easily integrated into both non-advanced and modern vehicles alike to enhance safety and visibility on the road.

## What it does
Blur Be Gone provides an integrated solution to reduce fogginess and increase visibility in real-time utilizing computer vision, image processing, and object recognition frameworks. Blur Be Safe prioritizes driver and civilian safety by serving as an embedded system that can be directly integrated into motor vehicles. Once fog is detected through the initial computer vision framework, the process then continues to the image processing and object-detection algorithms in real-time to provide visual enhancements that aid the driver's journey. 
## How we built it
Our python-based project has three primary computer vision components that contribute to the main functionality of the system. First, we have our fog detection algorithm, followed by various image enhancement algorithms, and finally we have our object-detection algorithms.

The fog detection algorithm utilized OpenCV to determine the presence of fog and then trigger the image enhancement processes based off of a foggy state. Once the system has entered a foggy state, the image processing algorithm begins and utilizes CLAHE, Dark Channel Prior, and additional sharpness/saturation filters to detect edges and increase image contrast in real time, providing visibility within adverse conditions. Following this, we enter into the final phase of our system, which is the object-detection framework that utilized YOLOv8 to detect and classify objects within the enhanced image frame in real-time. We tuned the confidence level of this pre-trained model in order to meet the specifications of our project.

We ran the models using PyQT5 to create a GUI that accessed the webcam data live and applied all necessary filters for our desired visual outputs.

## Challenges we ran into
Originally, we wanted to create a hardware and software-based solution utilizing a Raspberry Pi4 and camera module to directly emulate our innovative system. This ambitious idea was put on hold due to several networking restrictions, such as firewall configurations that prevented SSH and Virtual Network Computing.

After successfully overcoming the networking restrictions, we were able to flash our Pi4 with the relevant Python code and computer vision applications, however there were imaging latency and thermal issues due to the lack of processing capability on-board the Pi. We were unable to fully integrate this with our limited hardware and shifted directions to successfully implement our functioning software component.

Fortunately, we were then able to implement the desired functionality of the original hardware assembly through the native webcam of one of our devices and a GUI made using PyQT5.
## Accomplishments that we're proud of
Three members of our team were new to hackathons as a whole, and two were electrical engineering students with limited software development exposure. Having a fully completed project that we were able to demo at the completion of the event was a significant milestone for us to meet considering our background. Additionally, we are proud of the various computer vision aspects that we were able to implement and layer into one integrated system in order to produce a high-visibility output with object recognition.
## What we learned
As previously mentioned, two of our members were electrical engineering students with little exposure to computer vision applications and frameworks. This project provided a well-rounded introduction to AI/ML applications within the automotive industry, and more-specifically allowed for thorough experimentation with computer vision and various libraries in Python.
## What's next for BlurBeGone
As newer vehicle models advance technologically, older existing models have fallen behind. This solution will prove beneficial as an aftermarket addon for affordable assisted driving. Additionally, assisted driving systems employ multiple tools in addition to cameras, such as LiDAR and radar. An improvement in the capabilities of camera technologies uplifts modern systems as a whole. Utilizing LiDAR and radar will provide supplemental imaging support to assist in the image processing and capabilities of our current framework. Finally, cutting edge heads up displays provide an immersive way for the driver to receive information. The defogger can provide users with a live augmented view of their environment in real time and provide alerts in-vehicle that contribute to overall road safety and hazard management,

