import cv2 # type: ignore
import numpy as np # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image # type: ignore

def recognize_face(image_path=None, use_webcam=True):
    # Load model and label encoder
    model = load_model("models/face_model.h5")
    classes = np.load("models/label_encoder.npy", allow_pickle=True)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if use_webcam:
        cap = cv2.VideoCapture(0)
    else:
        frame = cv2.imread(image_path)
    
    while True:
        if use_webcam:
            ret, frame = cap.read()
            if not ret:
                break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_img = cv2.resize(face_img, (128, 128))
            face_img = face_img / 255.0
            face_img = face_img.reshape(1, 128, 128, 1)
            
            # Predict
            pred = model.predict(face_img)
            label_idx = np.argmax(pred)
            label = classes[label_idx]
            confidence = pred[0][label_idx]
            
            # Display result
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow('Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    if use_webcam:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_face()