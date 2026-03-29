from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# 🚨 لا تغيّر النموذج (حسب طلبك)
model = YOLO('yolo11n.pt')  # أو النموذج الذي تستخدمه

@app.route("/")
def home():
    return "YOLO API Running 🚀"

@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        image_base64 = data["image"]

        # إزالة header إذا موجود
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        # فك التشفير
        img_bytes = base64.b64decode(image_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        # YOLO inference (بدون تغيير النموذج)
        results = model(img)[0]

        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            detections.append({
                "label": model.names[cls],
                "confidence": round(conf, 3)
            })

        return jsonify({
            "detections": detections
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
