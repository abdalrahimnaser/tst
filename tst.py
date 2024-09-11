import cv2
import numpy as np
import time

# Load YOLO model
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load input from webcam or video (adjust as needed)
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with video path

# Set up FPS calculation
fps_start_time = 0
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    
    start_time = time.time()
    outs = net.forward(output_layers)
    fps = 1 / (time.time() - start_time)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Optional: Non-maximum suppression to reduce overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Print detected objects, confidence, and FPS
    for i in indexes:
        i = i[0]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        print(f"Object: {label}, Confidence: {confidence:.2f}, FPS: {fps:.2f}")

    # Break after processing (remove to keep running indefinitely)
    break

cap.release()
