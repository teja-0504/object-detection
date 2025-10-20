import cv2
import numpy as np
from ultralytics import YOLO

def main():
    # Load the YOLO model (pre-trained on COCO dataset)
    model = YOLO('yolov8n.pt')  # Using the nano model for speed

    # Open the default camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Perform object detection
        results = model(frame)

        # Render results on the frame
        annotated_frame = results[0].plot()

        # Display the frame
        cv2.imshow('Object Detection', annotated_frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
