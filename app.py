from flask import Flask, render_template, request, jsonify
import cv2 
import numpy as np
from prediction import predict
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST': 
        img = request.files['img']
        img = cv2.imdecode(np.fromstring(img.read(), np.uint8), cv2.IMREAD_COLOR)
        model_path = 'models/model.pkl'
        scaler_path = 'models/scaler.pkl'
        categories = ["blackhead", "whitehead", "pustula", "nodule"]

        label, confidence, feat = predict(img, model_path, scaler_path, categories)
        if label:
            return jsonify({
                'label': str(label),
                'confidence': str(confidence),
                'feature': str(feat)
            })
        else:
            return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
