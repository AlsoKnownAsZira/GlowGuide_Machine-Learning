from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import logging
from prediction import predict
from facedet import extract_face
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # get image from request
        img = request.files['img'].stream
        img = cv2.imdecode(np.fromstring(img.read(), np.uint8), cv2.IMREAD_COLOR) # Simpan gambar dari request
        model_path = 'models\model.pkl'
        scaler_path = 'models\scaler.pkl'
        categories = ["blackhead", "whitehead", "pustula", "nodule"]  # Sesuaikan dengan kategori Anda
        
        # extract face
        # face = extract_face(img)

        # make prediction
        label, confidence, feat = predict(img, model_path, scaler_path, categories)
        if label:
            return jsonify({
                'label': str(label),
                'confidence': str(confidence),
                'feature': str(feat)
            })
        else:
            return jsonify({'error': 'Prediction failed'}), 500
