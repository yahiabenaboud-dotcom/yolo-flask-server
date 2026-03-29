from ultralytics import YOLO
import cv2
import numpy as np
import base64

model = YOLO('yolov8n.pt')

def detect_objects(image_base64):
    img_bytes = base64.b64decode(image_base64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        detections.append({
            'label': label,
            'confidence': conf,
            'bbox': [x1, y1, x2, y2]
        })

    annotated_img = results.plot()
    _, buffer = cv2.imencode('.jpg', annotated_img)
    annotated_base64 = base64.b64encode(buffer).decode('utf-8')

    return detections, annotated_base64