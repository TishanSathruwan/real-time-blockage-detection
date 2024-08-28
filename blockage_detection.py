import cv2
import time
import numpy as np
from src.model import blockageDetectionModel

# video feed initiation
cap = cv2.VideoCapture(0)

# check the availability of the camera
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# parameter initiation
prev_time = time.time()
fps = 0

# model initiation
model = blockageDetectionModel()

while True:
    
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame was captured successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    fps_display = int(fps)

    # inference
    results = model.forward(frame)
    
    depth_frame = results[1]
    depth_frame = cv2.applyColorMap(np.uint8(depth_frame), cv2.COLORMAP_INFERNO)
    
    # Draw bounding boxes and create binary mask
    if len(results[0][0].boxes.xyxy.cpu().numpy()) > 0:
        for row in results[0][0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, row)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.rectangle(depth_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    # Display the FPS on the original frame
    cv2.putText(frame, f'FPS: {fps_display}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show depth estimated feed and detected feed
    cv2.imshow('Detection Feed', frame)
    cv2.imshow('Depth Estimated Feed', depth_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        captured_frame = frame
        break

cap.release()
cv2.destroyAllWindows()