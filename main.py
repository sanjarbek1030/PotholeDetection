"""
Pothole Detection Script
-------------------------
This script uses a pre-trained YOLOv8 model (best.pt) to detect potholes
in a video file (road_input.mp4). It displays the live detection window
and saves the annotated result to pothole_output.mp4.

Project structure expected (all in the same folder):
    - pothole_detection.py   (this script)
    - best.pt                (your trained YOLOv8 model)
    - road_input.mp4         (the video you want to process)

Run this in PyCharm with the 'ultralytics' and 'opencv-python' packages
installed:
    pip install ultralytics opencv-python
"""

import cv2
from ultralytics import YOLO

# -----------------------------------------------------------------------
# 1. LOAD THE MODEL
# -----------------------------------------------------------------------
# YOLO() loads our custom-trained pothole detection weights from 'best.pt'.
# This file must be in the same folder as this script (project root).
model = YOLO("pothole_model.pt")

# -----------------------------------------------------------------------
# 2. VIDEO CAPTURE - open the input video
# -----------------------------------------------------------------------
# cv2.VideoCapture opens the video file so we can read it frame by frame.
input_video_path = "road_input.mp4"
cap = cv2.VideoCapture(input_video_path)

# Always check that the video actually opened. If the path is wrong or
# the file is corrupted, this will fail, and we should stop early.
if not cap.isOpened():
    print(f"Error: Could not open video file '{input_video_path}'.")
    exit()

# Grab key properties from the input video so our output video matches:
# - fps: frames per second (controls playback speed)
# - frame_width / frame_height: resolution of each frame
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Input video loaded: {frame_width}x{frame_height} @ {fps:.2f} FPS")

# -----------------------------------------------------------------------
# 3. VIDEO WRITER - set up the output video file
# -----------------------------------------------------------------------
# cv2.VideoWriter saves our processed (annotated) frames into a new video.
# 'mp4v' is a common codec (FourCC code) used for writing .mp4 files.
output_video_path = "pothole_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# -----------------------------------------------------------------------
# 4. MAIN PROCESSING LOOP - read, detect, display, and save each frame
# -----------------------------------------------------------------------
print("Starting pothole detection... Press 'q' to stop early.")

while True:
    # cap.read() grabs the next frame from the video.
    # 'ret' is True if a frame was successfully read, False if the video ended.
    ret, frame = cap.read()

    if not ret:
        print("Finished processing video (no more frames).")
        break

    # ---- MODEL INFERENCE ----
    # Pass the current frame to the YOLO model. The model returns a list
    # of "Results" objects (one per frame) containing detected potholes:
    # their bounding boxes, confidence scores, and class labels.
    results = model(frame)

    # results[0].plot() automatically draws the bounding boxes, labels,
    # and confidence scores onto a copy of the frame for us — no manual
    # drawing code needed.
    annotated_frame = results[0].plot()

    # ---- DISPLAY THE LIVE PROCESSING WINDOW ----
    # Shows the annotated frame in a pop-up window in real time.
    cv2.imshow("Pothole Detection - Live", annotated_frame)

    # ---- SAVE THE FRAME TO THE OUTPUT VIDEO ----
    out.write(annotated_frame)

    # Wait 1 millisecond between frames and check if the user pressed 'q'
    # to quit early. cv2.waitKey is required for imshow() to refresh.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Processing stopped by user.")
        break

# -----------------------------------------------------------------------
# 5. CLEANUP - release resources properly
# -----------------------------------------------------------------------
# It's important to release the video capture and writer objects, and
# close any OpenCV windows, so files are saved correctly and no system
# resources (camera/file handles) are left open.
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Done! Processed video saved as '{output_video_path}'.")
