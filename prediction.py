import cv2
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern

# Preprocess image: resize and convert to L*a*b
def preprocess_image(image, size=(150, 150)):
    image = cv2.resize(image, size)
    return image

# Extract color features using L*a*b*, HSV, YCbCr, and Redness Index
def extract_color_features(image):
    # Convert to L*a*b*
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Convert to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv_image)
    
    # Convert to YCbCr
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_channel, cb_channel, cr_channel = cv2.split(ycbcr_image)

    # Redness Index (RI)
    r, g, b = cv2.split(image)
    redness_index = (r.astype(float) ** 2) / (g.astype(float) * b.astype(float) + 1e-6)  # Avoid division by zero
    mean_ri = np.mean(redness_index)
    var_ri = np.var(redness_index)

    # Extract mean and variance for L*a*b*, HSV, and YCbCr
    features = []
    for channel in [l_channel, a_channel, b_channel, h_channel, s_channel, v_channel, cb_channel, cr_channel]:
        features.append(np.mean(channel))  # Mean
        features.append(np.var(channel))  # Variance

    # Add Redness Index features
    features.extend([mean_ri, var_ri])
    
    return np.array(features)

# Extract texture features using LBP, Edge Detection, and Contour Analysis
def extract_texture_features(image):
    gray_image = rgb2gray(image)
    
    # Convert to uint8
    gray_image = (gray_image * 255).astype(np.uint8)
    
    # 1. Local Binary Pattern (LBP)
    lbp = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    lbp_hist = lbp_hist / lbp_hist.sum()  # Normalize histogram

    # 2. Edge Detection (Canny)
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
    edge_density = np.sum(edges) / edges.size  # Fraction of edges in the image

    # 3. Contour Analysis
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_features = [
        len(contours),  # Number of contours
        np.mean([cv2.contourArea(c) for c in contours]) if contours else 0,  # Mean contour area
        np.std([cv2.contourArea(c) for c in contours]) if contours else 0  # Std contour area
    ]

    # Combine all texture features
    return np.concatenate((lbp_hist, [edge_density], contour_features))

def load_model_and_scaler(model_path, scaler_path):
    try:
        model = joblib.load(model_path)  # Load model
        scaler = joblib.load(scaler_path)  # Load scaler
        return model, scaler
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return None, None
    
# Extract combined features: color and contour
def extract_combined_features(image):
    color_features = extract_color_features(image)
    contour_features = extract_texture_features(image)
    return np.concatenate((color_features, contour_features))
    
def predict(image, model_path, scaler_path, categories):
    try:
        # Load model and scaler
        model, scaler = load_model_and_scaler(model_path, scaler_path)
        if model is None or scaler is None:
            return None, None
        
        # Preprocess the image
        image = preprocess_image(image)
        
        # Extract combined features
        feature_vector = extract_combined_features(image)
        
        # Ensure consistent feature vector shape
        max_length = model.n_features_in_
        feature_vector = np.pad(feature_vector, (0, max_length - len(feature_vector)), constant_values=0)
        feature_vector = feature_vector.reshape(1, -1)
        
        # Scale the feature vector
        feature_vector_scaled = scaler.transform(feature_vector)
        
        # Predict using the trained model
        probabilities = model.predict_proba(feature_vector_scaled)
        predicted_label_idx = np.argmax(probabilities)
        confidence = probabilities[0][predicted_label_idx]
        
        # Return predicted label and confidence
        return categories[predicted_label_idx], confidence, feature_vector_scaled
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, None, None