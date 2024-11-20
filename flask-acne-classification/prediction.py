from hpelm import ELM
from skimage.feature import graycomatrix, graycoprops
import cv2
import numpy as np
import joblib

def prepocess(img):
    std_img = cv2.resize(img, (150,150))
    img_bgr = cv2.cvtColor(std_img, cv2.COLOR_RGB2BGR)  # OpenCV uses BGR format
    lab_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)

    return lab_img

def kmeans_segmentation(image, clusters=2):
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    lesion_mask = (labels.flatten() == labels[0]).astype(np.uint8) * 255
    binary_image = lesion_mask.reshape((image.shape[0], image.shape[1]))

    mask = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
    masked_segmented_image = cv2.bitwise_and(cv2.cvtColor(image, cv2.COLOR_Lab2RGB), mask)

    return binary_image, masked_segmented_image

def extract_glcm_features(masked_segmented_image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    gray_image = cv2.cvtColor(masked_segmented_image, cv2.COLOR_BGR2GRAY)
    
    glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

    features = {
        'contrast': graycoprops(glcm, 'contrast').mean(),
        'homogeneity': graycoprops(glcm, 'homogeneity').mean(),
        'energy': graycoprops(glcm, 'energy').mean(),
        'correlation': graycoprops(glcm, 'correlation').mean(),
    }

    return features

def extract_feat(img):
    _, segmented_img = kmeans_segmentation(img)
    features = extract_glcm_features(segmented_img)

    return features

def predict(img):
    elm_model = ELM(4, 5)
    elm_model.load('models/elm_model.h5')
    labels = joblib.load('models/acne_labels.pkl')

    # preprocessing input image
    img = prepocess(img)
    feat = extract_feat(img)
    X = np.array([[feat['contrast'], feat['correlation'], feat['energy'], feat['homogeneity']]])

    # make prediction
    pred = elm_model.predict(X)
    pred_conf = np.amax(pred)
    pred_index = np.argmax(pred)
    pred_label = labels[pred_index]
    
    return pred_label, pred_conf