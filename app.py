from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys

app = Flask(__name__)
CORS(app)  # السماح لجميع النطاقات بالاتصال (ضروري لتطبيق Flutter)

# تحميل النموذج (استخدم yolov8n.pt لأنه أخف وزناً)
print("Loading YOLO model...", file=sys.stderr)
try:
    model = YOLO('yolov8n.pt')
    print("Model loaded successfully", file=sys.stderr)
except Exception as e:
    print(f"Error loading model: {e}", file=sys.stderr)
    sys.exit(1)

@app.route('/detect', methods=['POST'])
def detect():
    # استقبال الصورة بصيغة base64
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    image_base64 = data['image']
    # إزالة أي رأس (مثل "data:image/jpeg;base64,") إذا كان موجوداً
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]

    try:
        # فك base64 إلى صورة
        img_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # كشف الأشياء
        results = model(img)[0]
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            detections.append({'label': label, 'confidence': conf})

        # إرجاع النتائج
        return jsonify({'detections': detections})

    except Exception as e:
        print(f"Error during detection: {e}", file=sys.stderr)
        return jsonify({'error': str(e)}), 500

# مسار بسيط للتحقق من أن الخادم يعمل (GET)
@app.route('/ping', methods=['GET'])
def ping():
    return "OK", 200

if __name__ == '__main__':
    # استخدام المنفذ المحدد من Railway أو 5000 افتراضياً
    port = int(os.environ.get('PORT', 5000))
    # تشغيل الخادم على جميع الواجهات
    app.run(host='0.0.0.0', port=port, debug=False)
