import numpy as np
import pickle
import os
import cv2 # You will need OpenCV for image processing
import dlib # dlib is needed for landmark extraction
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from face_feature_extractor import landmarks_to_feature_vector

# --- Configuration ---
MODEL_OUTPUT_PATH = "face_shape_classifier.pkl"
# You MUST download the shape predictor file for this to work
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# --- Helper function for landmark extraction ---
def get_landmarks(image, detector, predictor):
    """Extracts 68 landmarks from a single face in an image."""
    if image is None:
        return None
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(img_gray, 1)
    if len(faces) == 0:
        return None
    # Assume the first face found is the one we want
    shape = predictor(img_gray, faces[0])
    return [(shape.part(i).x, shape.part(i).y) for i in range(68)]

def load_real_data(dataset_path, detector, predictor):
    """
    Loads a real dataset of images, extracts landmarks, and converts them to feature vectors.

    The dataset should be organized in folders, where each folder is named after the face shape.
    Example:
    - dataset/
        - Round/
            - image1.jpg
            - image2.png
        - Oval/
            - image3.jpg
    """
    print("Loading real data from:", dataset_path)
    features_list = []
    labels = []
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset path not found: {dataset_path}")
        return np.array([]), np.array([])

    for shape_label in os.listdir(dataset_path):
        shape_dir = os.path.join(dataset_path, shape_label)
        if os.path.isdir(shape_dir):
            for filename in os.listdir(shape_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(shape_dir, filename)
                    image = cv2.imread(image_path)
                    
                    landmarks = get_landmarks(image, detector, predictor)
                    
                    if landmarks:
                        features = landmarks_to_feature_vector(landmarks)
                        if features.size > 0:
                            features_list.append(features)
                            labels.append(shape_label)
    
    print(f"Successfully loaded and processed {len(features_list)} images.")
    return np.array(features_list), np.array(labels)

def train_classifier(X, y):
    """Trains a RandomForest Classifier on the features."""
    if X.shape[0] < 2:
        print("Error: Not enough data to train. You need at least 2 samples.")
        return None
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Increased estimators and depth for a more robust model
    model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Training Complete.")
    print(f"Test Accuracy on Real Data: {accuracy*100:.2f}%")
    
    return model

def save_model(model):
    """Saves the trained model to a pickle file."""
    if model:
        with open(MODEL_OUTPUT_PATH, 'wb') as file:
            pickle.dump(model, file)
        print(f"Trained model saved to: {MODEL_OUTPUT_PATH}")

if __name__ == '__main__':
    # Initialize dlib's face detector and shape predictor
    if not os.path.exists(SHAPE_PREDICTOR_PATH):
        print("="*50)
        print(f"FATAL ERROR: Dlib shape predictor file not found!")
        print(f"Please download '{SHAPE_PREDICTOR_PATH}' and place it in the same folder.")
        print("You can get it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("="*50)
    else:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        
        # --- IMPORTANT ---
        # 1. Create a folder named 'dataset' in your project directory.
        # 2. Inside 'dataset', create subfolders for each face shape: 'Round', 'Oval', etc.
        # 3. Place your labeled images into the correct folders.
        # 4. Then run this script.
        X, y = load_real_data('dataset', detector, predictor)
        
        if X.size > 0:
            trained_model = train_classifier(X, y)
            save_model(trained_model)
        else:
            print("Could not train the model because no data was loaded.")