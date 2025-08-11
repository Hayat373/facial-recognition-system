import cv2
import os
from ultralytics import YOLO

def capture_objects(object_name, data_dir="data/training"):
      # Create directory for the object's images
      object_dir = os.path.join(data_dir, object_name)
      os.makedirs(object_dir, exist_ok=True)
      
      # Load YOLO model
      model = YOLO("yolov8n.pt")  # Pre-trained YOLOv8 nano model
      
      # Initialize webcam
      cap = cv2.VideoCapture(0)
      
      count = 0
      max_images = 100  # Number of images to capture
      
      print(f"Capturing images for {object_name}. Press 'q' to stop early.")
      
      while count < max_images:
          ret, frame = cap.read()
          if not ret:
              break
          
          # Detect objects using YOLO
          results = model(frame)
          
          for result in results:
              boxes = result.boxes.xyxy  # Bounding boxes
              classes = result.boxes.cls  # Class IDs
              names = result.names  # Class names
              
              for box, cls in zip(boxes, classes):
                  class_name = names[int(cls)]
                  if class_name.lower() == object_name.lower():  # Match user-specified object
                      x1, y1, x2, y2 = map(int, box)
                      # Save cropped object
                      object_img = frame[y1:y2, x1:x2]
                      img_path = os.path.join(object_dir, f"{object_name}_{count}.jpg")
                      cv2.imwrite(img_path, object_img)
                      count += 1
                      
                      # Draw rectangle around object
                      cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                      cv2.putText(frame, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
          
          # Display frame
          cv2.imshow('Capturing Objects', frame)
          
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
      
      cap.release()
      cv2.destroyAllWindows()
      print(f"Captured {count} images for {object_name}.")

if __name__ == "__main__":
      object_name = input("Enter the object name (e.g., car, chair): ")
      capture_objects(object_name)