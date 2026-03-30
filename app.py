from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
from ultralytics import YOLO
import os

app = Flask(__name__)
CORS(app)

model = YOLO('yolo11n.pt')

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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
