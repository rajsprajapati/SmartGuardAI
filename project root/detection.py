import os
import cv2

def detect_objects_in_video(video_file, output_file, output_directory, modelFile, configFile, classFile):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_path = os.path.join(output_directory, output_file)
    
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    with open(classFile, 'r') as f:
        labels = f.read().strip().split('\n')

    FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    THICKNESS = 1

    def detect_objects(net, im):
        dim = 300
        blob = cv2.dnn.blobFromImage(im, 1.0, size=(dim, dim), mean=(0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        objects = net.forward()
        return objects

    def display_text(im, text, x, y):
        textSize = cv2.getTextSize(text, FONTFACE, FONT_SCALE, THICKNESS)
        dim = textSize[0]
        baseline = textSize[1]
        cv2.rectangle(im, (x, y - dim[1] - baseline), (x + dim[0], y + baseline), (0, 0, 0), cv2.FILLED)
        cv2.putText(im, text, (x, y - 5), FONTFACE, FONT_SCALE, (0, 255, 255), THICKNESS, cv2.LINE_AA)

    def display_objects(im, objects, threshold=0.25):
        rows, cols, _ = im.shape
        for i in range(objects.shape[2]):
            classId = int(objects[0, 0, i, 1])
            score = float(objects[0, 0, i, 2])
            if score > threshold:
                x, y, x1, y1 = map(int, objects[0, 0, i, 3:7] * [cols, rows, cols, rows])
                w, h = x1 - x, y1 - y
                display_text(im, "{}".format(labels[classId]), x, y)
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        objects = detect_objects(net, frame)
        display_objects(frame, objects)
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Detected video saved as {output_path}")
    return True
