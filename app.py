from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# ==================================================
# استخدام نموذج أقوى وأدق:
# - yolo11n.pt  (أحدث إصدار، دقة عالية وسريع)
# - yolov8s.pt  (أكبر من yolov8n، دقة أفضل)
# - yolo26n.pt  (أحدث إصدار تجريبي، يتطلب تحديث ultralytics)
# ==================================================
# اختر أحد السطور التالية (افتح التعليق عن النموذج المطلوب)
model = YOLO('yolo11n.pt')      # يوصى به (يتنزّل تلقائياً أول مرة)
# model = YOLO('yolov8s.pt')    # دقة أعلى من yolov8n
# model = YOLO('yolo26n.pt')    # أحدث نموذج (جرب إذا كنت تريد الأحدث)

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    image_base64 = data['image']
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]

    img_bytes = base64.b64decode(image_base64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)[0]
    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])
        detections.append({'label': label, 'confidence': conf})

    return jsonify({'detections': detections})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)