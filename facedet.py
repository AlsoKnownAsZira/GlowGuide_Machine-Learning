import cv2

def extract_face(image, output_path=None):
    # Load Haar Cascade untuk deteksi wajah
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    
    # Konversi ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(150, 150))

    if len(faces) == 0:
        raise ValueError("No faces detected")

    # Ekstraksi wajah
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        if output_path:
            cv2.imwrite(output_path, face)
        return face

    raise ValueError("Failed to extract face")