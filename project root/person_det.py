import numpy as np
import cv2
import time
import os
import psycopg2
from datetime import datetime, date

# Load the pre-trained SSD MobileNet model for person detection
modelFile = './project/demo_work/models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'
configFile = './project/demo_work/models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
classFile = './project/demo_work/coco_class_labels.txt'

# Load the COCO class labels
with open(classFile, 'r') as f:
    class_labels = f.read().strip().split('\n')

# Get the index of the 'person' class (class index 0 in COCO dataset)
person_class_id = class_labels.index('person')

# Load the SSD model
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

# Initialize counters and trackers
person_count = 0
person_tracker = {}
person_last_detected = {}

# Define a threshold distance to consider a person the same
threshold_distance = 50

# Set the path to the static folder for saving images

image_save_dir = '/project/project root/static/images/'
os.makedirs(image_save_dir, exist_ok=True)

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="person_detection",
    user="postgres",  # Replace with your PostgreSQL username
    password="raj123456",  # Replace with your PostgreSQL password
    host="localhost"
)
cursor = conn.cursor()

# Function to calculate centroid
def calculate_centroid(x, y, x1, y1):
    return (x + (x1 - x) // 2, y + (y1 - y) // 2)

# Person detection function to be called by the Flask app
def detect_persons(frame, session_id):
    global person_count, person_tracker, person_last_detected

    current_time = time.time()
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    current_frame_centroids = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])

        if confidence > 0.5 and class_id == person_class_id:  # Only detect people
            # Get the coordinates of the detected person
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x, y, x1, y1) = box.astype("int")

            # Calculate the centroid of the detected person
            centroid = calculate_centroid(x, y, x1, y1)
            current_frame_centroids.append((centroid, (x, y, x1, y1)))

            # Assign ID to the detected person
            person_id = None
            for pid, prev_centroid in person_tracker.items():
                distance = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
                if distance < threshold_distance:
                    person_id = pid
                    person_tracker[person_id] = centroid  # Update position
                    break

            # If no match found, assign a new person ID
            if person_id is None:
                person_count += 1
                person_id = person_count
                person_tracker[person_id] = centroid
                person_last_detected[person_id] = current_time
                print(f"New person detected! Assigned ID: {person_id}")

            # If 2 seconds have passed since detection, save the image and store in DB
            if current_time - person_last_detected[person_id] >= 2:
                print(f"Capturing image for person {person_id}")
                image_name = f'person_{person_id}_{x}_{y}.jpg'
                image_path = os.path.join(image_save_dir, image_name)
                cv2.imwrite(image_path, frame[y:y1, x:x1])

                # Insert detection data into PostgreSQL
                now = datetime.now()
                detection_time = now.strftime("%H:%M:%S")
                detection_date = date.today()
                # cursor.execute(
                #     "INSERT INTO detected_person (id, person_id, image_path, detection_time, detection_date, image_name) VALUES (%s, %s, %s, %s, %s, %s)",
                #     (1, person_id, image_path, detection_time, detection_date, image_name)
                # )
                cursor.execute(
                    "INSERT INTO detected_person (id, person_id, image_path, detection_time, detection_date, image_name) VALUES (%s, %s, %s, %s, %s, %s)",
                    (session_id, person_id, image_path, detection_time, detection_date, image_name)
                )
                conn.commit()
                print(f"Detection data stored in database for person {person_id}")
                person_last_detected[person_id] = float('inf')

            # Draw rectangle and label
            (x, y, x1, y1) = current_frame_centroids[-1][1]
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame, f'Person {person_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame
