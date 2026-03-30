from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

app = Flask(__name__)
CORS(app)

# استخدام نموذج أصغر حجماً وأقل استهلاكاً للذاكرة
print("Loading YOLO model...")
model = YOLO('yolov8n.pt')   # ~6 MB only
print("Model loaded successfully")

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    image_base64 = data['image']
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]

    try:
        img_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # الكشف
        results = model(img)[0]
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            detections.append({'label': label, 'confidence': conf})

        return jsonify({'detections': detections})
    except Exception as e:
        print(f"Error during detection: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
