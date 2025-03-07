import cv2
from person_det import detect_persons
import time

# Global variable to track status
camera_running = True



# Background processing function (detection runs without displaying the video)
def process_frames(source, session_id):
    global camera_running
    if source:
        # Check if the source is an integer (camera index) or a URL
        try:
            source = int(source)  # Try to cast to int for camera index
        except ValueError:
            pass  # If it fails, it's likely a URL
        
        camera = cv2.VideoCapture(source)
        while camera_running:
            with open('status.txt', 'r') as file:
                print(camera_running)
                camera_running = file.read().strip() == 'True'
                print(camera_running)
                # Read the camera frame
                success, frame = camera.read()
                print(camera_running)
                if not camera_running:
                    print("Camera stopped.")
                    break
                else:
                    if not success:
                        break
                    # Perform person detection on the frame (no display/output here)
                    detect_persons(frame, session_id)
                    print("Camera running...")
                    time.sleep(1)
    else:
        return "Invalid input"


import threading

def generate_frames(source, session_id):
    global camera_running
    if source:
        try:
            source = int(source)  # Try to use it as a camera index if it's an integer
        except ValueError:
            pass  # If not, it's likely a URL

        camera = cv2.VideoCapture(source)
        # camera = cv2.VideoCapture(source, cv2.CAP_DSHOW)

        while camera_running:
            with open('status.txt', 'r') as file:
                print(camera_running)
                camera_running = file.read().strip() == 'True'
                print(camera_running)
                success, frame = camera.read()
                if not camera_running or not success:
                    print(camera_running)
                    break
                else:
                    frame = cv2.resize(frame, (640, 480))  # Resize frame to 640x480
                    thread = threading.Thread(target=detect_persons, args=(frame,  session_id))
                    thread.start()
                    thread.join()
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    time.sleep(0.1)  # Reduce sleep time to increase frame rate
    else:
        return "Invalid input"


