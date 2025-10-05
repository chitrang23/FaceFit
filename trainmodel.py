import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Import the feature extraction logic
from face_feature_extractor import landmarks_to_feature_vector 

# --- Configuration ---
MODEL_OUTPUT_PATH = "face_shape_classifier.pkl"

def load_and_preprocess_data():
    """
    MOCK FUNCTION: Generates a synthetic dataset for demonstration purposes.
    
    !!! IMPORTANT: In a production application, you MUST replace this function
    with code that loads real, manually labeled images/landmark data.
    """
    print("Generating synthetic data. Replace this with real dataset loading!")
    num_samples = 200
    features_list = []
    labels = []
    face_shapes = ['Round', 'Oval', 'Square', 'Heart', 'Long']
    
    for i in range(num_samples):
        # Generate randomized mock landmarks 
        mock_landmarks = [(np.random.randint(100, 400) + np.random.normal(0, 5), 
                           np.random.randint(100, 400) + np.random.normal(0, 5)) 
                          for _ in range(68)]
        
        mock_landmarks = [(int(x), int(y)) for x, y in mock_landmarks]
        
        # Extract features using the dedicated module
        features = landmarks_to_feature_vector(mock_landmarks)
        
        if features.size > 0:
            features_list.append(features)
            # Assign shape cyclically for balanced mock data
            labels.append(face_shapes[i % len(face_shapes)])

    print(f"Generated {len(features_list)} feature vectors for training.")
    return np.array(features_list), np.array(labels)

def train_classifier(X, y):
    """Trains a RandomForest Classifier on the features."""
    
    if len(X) == 0:
        print("Error: No features found for training.")
        return None
        
    # Split data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Training Complete.")
    print(f"Test Accuracy: {accuracy*100:.2f}% (Note: Accuracy on synthetic data is not meaningful)")
    
    return model

def save_model(model):
    """Saves the trained model to a pickle file."""
    if model is None:
        return
        
    try:
        with open(MODEL_OUTPUT_PATH, 'wb') as file:
            pickle.dump(model, file)
        print(f"Trained model successfully saved to: {MODEL_OUTPUT_PATH}")
    except Exception as e:
        print(f"Error saving model: {e}")

# --- Main Execution ---
if __name__ == '__main__':
    print("Starting Face Shape Classifier Training...")
    
    # 1. Load Data and Extract Features
    X, y = load_and_preprocess_data()
    
    # 2. Train Model
    trained_model = train_classifier(X, y)
    
    # 3. Save Model
    save_model(trained_model)
