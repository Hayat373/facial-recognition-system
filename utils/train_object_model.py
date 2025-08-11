import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder

def load_data(data_dir="data/training"):
    images = []
    labels = []
    
    for obj in os.listdir(data_dir):
        obj_dir = os.path.join(data_dir, obj)
        if os.path.isdir(obj_dir):
            for img_name in os.listdir(obj_dir):
                img_path = os.path.join(obj_dir, img_name)
                img = load_img(img_path, target_size=(128, 128), color_mode='rgb')  # Use RGB for objects
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(obj)
    
    return np.array(images), np.array(labels)

def train_model():
    # Load data
    X, y = load_data()
    if len(X) == 0:
        print("No training data found.")
        return
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    # Build model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),  # RGB input
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train model
    model.fit(X, y_encoded, epochs=20, batch_size=32, validation_split=0.2)
    
    # Save model and label encoder
    model.save("models/object_model.h5")
    np.save("models/label_encoder.npy", le.classes_)
    print("Model trained and saved.")

if __name__ == "__main__":
    train_model()