#importing necessary libraries
import numpy as np
import cv2
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from skimage.feature import hog
from glob import glob
import os

# Function to load images and extract HOG features
def load_data(img_size=(128,128)):
    dataset_path = "dataset"
    images = []
    labels = []
    class_names = os.listdir(dataset_path)
    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)
        for image_path in glob(os.path.join(class_path, "*.jpg")):
            image = cv2.imread(image_path)
            if image is None:
                continue
            image = cv2.resize(image, img_size)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            features = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
            images.append(features)
            labels.append(class_name)
    return np.array(images), np.array(labels)

# Function to train the model and save both the model and label encoder
def train(x, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=45)
    model.fit(x_train, y_train)
    
    prediction = model.predict(x_test)
    accuracy = accuracy_score(y_test, prediction)
    
    # Save model and label encoder
    joblib.dump(model, "flower_classifier.pkl")
    joblib.dump(le, "label_encoder.pkl")
    
    print(f"Model trained with accuracy: {accuracy*100:.2f}%")

# Run the training
x, y = load_data()
train(x, y)
