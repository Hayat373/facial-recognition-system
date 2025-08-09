import cv2 # type: ignore
import os

def capture_faces(name, data_dir="data/training"):
    # Create directory for the person's images
    person_dir = os.path.join(data_dir, name)
    os.makedirs(person_dir, exist_ok=True)
    
    # Initialize webcam and face detector
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    count = 0
    max_images = 100  # Number of images to capture
    
    print(f"Capturing images for {name}. Press 'q' to stop early.")
    
    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale and detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            # Save cropped face
            face_img = frame[y:y+h, x:x+w]
            img_path = os.path.join(person_dir, f"{name}_{count}.jpg")
            cv2.imwrite(img_path, face_img)
            count += 1
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Capturing Faces', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {count} images for {name}.")

if __name__ == "__main__":
    name = input("Enter the person's name: ")
    capture_faces(name)