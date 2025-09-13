import cv2
import numpy as np
import dlib
import math
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import os

class AIAccessorySuggestor:
    def __init__(self):
        # Initialize face detector
        self.detector = dlib.get_frontal_face_detector()
        
        # Try to load the shape predictor
        try:
            self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            self.landmark_available = True
            print("Facial landmark detector loaded successfully.")
        except:
            print("Warning: Could not load shape predictor. Using fallback landmark estimation.")
            self.landmark_available = False
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(-1)
            if not self.cap.isOpened():
                raise ValueError("Could not open any camera")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # ML Model for accessory suggestion
        self.scaler = StandardScaler()
        self.knn_model = KNeighborsClassifier(n_neighbors=3)
        self.is_trained = False
        
        # Feature database
        self.feature_db = []
        self.accessory_db = []
        
        # Current state
        self.current_mode = 'analysis'  # 'analysis' or 'suggestion'
        self.suggested_accessories = []
        self.face_features = {}
        
        # Sample training data
        self.sample_training_data()
        
        print("AI Accessory Suggestor initialized successfully!")
        print("Press 'a' for analysis, 's' for suggestions, 't' to train model, 'q' to quit")
    
    def detect_faces(self, frame):
        """Detect faces in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        return faces
    
    def sample_training_data(self):
        """Create sample training data for demonstration"""
        # Sample features: [face_shape, skin_tone, eye_size, face_width/height_ratio]
        # Face shape: 0=oval, 1=round, 2=square, 3=heart
        # Skin tone: 0=light, 1=medium, 2=dark
        # Eye size: 0=small, 1=medium, 2=large
        
        # Training features
        sample_features = [
            [0, 0, 1, 0.7], [0, 1, 2, 0.7], [0, 2, 1, 0.7],  # Oval faces
            [1, 0, 0, 1.0], [1, 1, 1, 1.0], [1, 2, 2, 1.0],  # Round faces
            [2, 0, 2, 0.8], [2, 1, 1, 0.8], [2, 2, 0, 0.8],  # Square faces
            [3, 0, 2, 0.6], [3, 1, 1, 0.6], [3, 2, 0, 0.6],  # Heart faces
        ]
        
        # Corresponding accessory suggestions
        sample_accessories = [
            ['Aviator glasses', 'Dangling earrings'],  # Oval faces
            ['Angular glasses', 'Hoop earrings'],      # Round faces
            ['Round glasses', 'Stud earrings'],        # Square faces
            ['Cat-eye glasses', 'Long necklace'],      # Heart faces
            ['Rectangular glasses', 'Short necklace'], # Oval faces
            ['Round glasses', 'Choker'],              # Round faces
            ['Square glasses', 'Statement earrings'],  # Square faces
            ['Aviator glasses', 'Delicate necklace'],  # Heart faces
            ['Wayfarer glasses', 'Drop earrings'],    # Oval faces
            ['Browline glasses', 'Hoop earrings'],    # Round faces
            ['Round glasses', 'Stud earrings'],        # Square faces
            ['Cat-eye glasses', 'Long necklace'],      # Heart faces
        ]
        
        self.feature_db = sample_features
        self.accessory_db = sample_accessories
        
        # Train the model with sample data
        self.train_model()
    
    def extract_facial_features(self, frame, face):
        """Extract facial features for analysis"""
        landmarks = self.get_facial_landmarks(frame, face)
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        
        features = {}
        
        # 1. Determine face shape
        features['face_shape'] = self.determine_face_shape(landmarks, w, h)
        
        # 2. Extract skin tone
        features['skin_tone'] = self.extract_skin_tone(frame, face)
        
        # 3. Analyze eye size
        features['eye_size'] = self.analyze_eye_size(landmarks, w, h)
        
        # 4. Calculate face width/height ratio
        features['face_ratio'] = w / h
        
        # 5. Analyze facial symmetry
        features['symmetry'] = self.analyze_symmetry(landmarks)
        
        self.face_features = features
        return features
    
    def determine_face_shape(self, landmarks, face_width, face_height):
        """Determine face shape using ML approach"""
        if len(landmarks) < 16:
            return 0  # Default to oval
            
        # Calculate face shape metrics
        jaw_width = landmarks[16][0] - landmarks[0][0]
        cheekbone_width = landmarks[14][0] - landmarks[2][0]
        forehead_width = landmarks[13][0] - landmarks[3][0]
        face_length = landmarks[8][1] - landmarks[27][1]
        
        # Calculate ratios
        jaw_ratio = jaw_width / face_width
        cheekbone_ratio = cheekbone_width / face_width
        forehead_ratio = forehead_width / face_width
        length_ratio = face_length / face_height
        
        # Simple heuristic for face shape determination
        if abs(jaw_ratio - cheekbone_ratio) < 0.1 and abs(cheekbone_ratio - forehead_ratio) < 0.1:
            return 1  # Round
        elif jaw_ratio > cheekbone_ratio and jaw_ratio > forehead_ratio:
            return 2  # Square
        elif forehead_ratio > cheekbone_ratio and cheekbone_ratio > jaw_ratio:
            return 3  # Heart
        else:
            return 0  # Oval
    
    def extract_skin_tone(self, frame, face):
        """Extract skin tone from face region"""
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        
        # Extract face region
        face_region = frame[y:y+h, x:x+w]
        
        if face_region.size == 0:
            return 1  # Default to medium
            
        # Convert to HSV and extract value channel
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        values = hsv[:, :, 2].flatten()
        
        # Classify skin tone based on value
        avg_value = np.mean(values)
        
        if avg_value < 85:
            return 2  # Dark
        elif avg_value < 170:
            return 1  # Medium
        else:
            return 0  # Light
    
    def analyze_eye_size(self, landmarks, face_width, face_height):
        """Analyze eye size relative to face"""
        if len(landmarks) < 48:
            return 1  # Default to medium
            
        # Calculate eye width
        left_eye_width = landmarks[39][0] - landmarks[36][0]
        right_eye_width = landmarks[45][0] - landmarks[42][0]
        avg_eye_width = (left_eye_width + right_eye_width) / 2
        
        # Calculate eye height
        left_eye_height = landmarks[41][1] - landmarks[37][1]
        right_eye_height = landmarks[47][1] - landmarks[43][1]
        avg_eye_height = (left_eye_height + right_eye_height) / 2
        
        # Calculate eye size relative to face
        eye_size_ratio = (avg_eye_width * avg_eye_height) / (face_width * face_height)
        
        if eye_size_ratio < 0.0015:
            return 0  # Small
        elif eye_size_ratio < 0.0025:
            return 1  # Medium
        else:
            return 2  # Large
    
    def analyze_symmetry(self, landmarks):
        """Analyze facial symmetry"""
        if len(landmarks) < 68:
            return 0.5  # Default symmetry
            
        # Calculate symmetry between left and right sides
        left_points = landmarks[0:17]  # Jawline
        right_points = landmarks[16:33]
        right_points = right_points[::-1]  # Reverse to match left points
        
        symmetry_score = 0
        for i in range(min(len(left_points), len(right_points))):
            # Calculate distance between corresponding points
            dist = math.sqrt((left_points[i][0] - right_points[i][0])**2 + 
                             (left_points[i][1] - right_points[i][1])**2)
            # Normalize by face width (approximate)
            if i == 0 or i == len(left_points)-1:
                face_width = abs(left_points[0][0] - right_points[-1][0])
                if face_width > 0:
                    symmetry_score += dist / face_width
        
        # Average and invert (lower distance = higher symmetry)
        symmetry_score = 1 - (symmetry_score / min(len(left_points), len(right_points)))
        return symmetry_score
    
    def train_model(self):
        """Train the KNN model with sample data"""
        if len(self.feature_db) == 0:
            print("No training data available!")
            return False
            
        # Prepare features and labels
        X = np.array(self.feature_db)
        y = list(range(len(self.accessory_db)))  # Using indices as labels
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train the model
        self.knn_model.fit(X_scaled, y)
        self.is_trained = True
        
        print("Model trained successfully with", len(self.feature_db), "samples")
        return True
    
    def suggest_accessories(self, features):
        """Suggest accessories based on facial features"""
        if not self.is_trained:
            print("Model not trained yet!")
            return []
            
        # Prepare feature vector
        feature_vector = [
            features['face_shape'],
            features['skin_tone'],
            features['eye_size'],
            features['face_ratio'],
            features['symmetry']
        ]
        
        # Ensure feature vector matches training dimension
        if len(feature_vector) < len(self.feature_db[0]):
            # Pad with zeros if needed
            feature_vector.extend([0] * (len(self.feature_db[0]) - len(feature_vector)))
        elif len(feature_vector) > len(self.feature_db[0]):
            # Truncate if needed
            feature_vector = feature_vector[:len(self.feature_db[0])]
        
        # Scale features
        feature_vector = self.scaler.transform([feature_vector])
        
        # Get predictions
        distances, indices = self.knn_model.kneighbors(feature_vector)
        
        # Get suggested accessories
        suggestions = []
        for idx in indices[0]:
            if idx < len(self.accessory_db):
                suggestions.extend(self.accessory_db[idx])
        
        # Remove duplicates and return
        return list(dict.fromkeys(suggestions))
    
    def get_facial_landmarks(self, frame, face):
        """Get facial landmarks"""
        if self.landmark_available:
            try:
                landmarks = self.predictor(frame, face)
                return [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
            except:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                return self.estimate_landmarks(x, y, w, h)
        else:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            return self.estimate_landmarks(x, y, w, h)
    
    def estimate_landmarks(self, x, y, w, h):
        """Estimate facial landmarks"""
        landmarks = []
        
        # Jawline
        for i in range(17):
            px = x + (i * w) // 16
            py = y + h - 10
            landmarks.append((px, py))
        
        # Eyebrows
        for i in range(5):
            landmarks.append((x + w//4 - i*w//16, y + h//4))
        for i in range(5):
            landmarks.append((x + 3*w//4 - i*w//16, y + h//4))
        
        # Nose
        for i in range(9):
            landmarks.append((x + w//2, y + h//2 + i*h//16))
        
        # Eyes
        for i in range(6):
            angle = 2 * math.pi * i / 6
            landmarks.append((int(x + w//4 + 15*math.cos(angle)), int(y + h//3 + 10*math.sin(angle))))
        for i in range(6):
            angle = 2 * math.pi * i / 6
            landmarks.append((int(x + 3*w//4 + 15*math.cos(angle)), int(y + h//3 + 10*math.sin(angle))))
        
        # Mouth
        for i in range(20):
            angle = math.pi * i / 19
            landmarks.append((int(x + w//2 + 20*math.cos(angle)), int(y + 2*h//3 + 10*math.sin(angle))))
        
        return landmarks
    
    def visualize_analysis(self, frame, face, features):
        """Visualize facial analysis on the frame"""
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        
        # Draw face bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw landmarks
        landmarks = self.get_facial_landmarks(frame, face)
        for point in landmarks:
            cv2.circle(frame, point, 2, (0, 0, 255), -1)
        
        # Add analysis text
        face_shapes = ["Oval", "Round", "Square", "Heart"]
        skin_tones = ["Light", "Medium", "Dark"]
        eye_sizes = ["Small", "Medium", "Large"]
        
        y_offset = y - 150 if y > 150 else y + h + 20
        cv2.putText(frame, f"Face Shape: {face_shapes[features['face_shape']]}", 
                    (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Skin Tone: {skin_tones[features['skin_tone']]}", 
                    (x, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Eye Size: {eye_sizes[features['eye_size']]}", 
                    (x, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Face Ratio: {features['face_ratio']:.2f}", 
                    (x, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Symmetry: {features['symmetry']:.2f}", 
                    (x, y_offset + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def visualize_suggestions(self, frame, face, suggestions):
        """Visualize accessory suggestions on the frame"""
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        
        # Draw suggestions box
        suggestion_box_height = 100 + len(suggestions) * 25
        cv2.rectangle(frame, (10, 10), (400, suggestion_box_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, suggestion_box_height), (0, 255, 0), 2)
        
        # Add title
        cv2.putText(frame, "AI Suggested Accessories:", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add suggestions
        for i, suggestion in enumerate(suggestions):
            cv2.putText(frame, f"{i+1}. {suggestion}", (20, 65 + i*25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        """Main loop"""
        print("Starting AI Accessory Suggestor. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                frame = cv2.flip(frame, 1)
                display_frame = frame.copy()
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Process each face
                for face in faces:
                    if self.current_mode == 'analysis':
                        # Extract and display facial features
                        features = self.extract_facial_features(frame, face)
                        display_frame = self.visualize_analysis(display_frame, face, features)
                    
                    elif self.current_mode == 'suggestion':
                        # Extract features and suggest accessories
                        features = self.extract_facial_features(frame, face)
                        suggestions = self.suggest_accessories(features)
                        self.suggested_accessories = suggestions
                        display_frame = self.visualize_suggestions(display_frame, face, suggestions)
                
                # Display mode info
                cv2.putText(display_frame, f"Mode: {self.current_mode.upper()}", 
                            (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press a=analysis, s=suggestions, t=train, q=quit", 
                            (10, display_frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('AI Accessory Suggestor', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('a'):
                    self.current_mode = 'analysis'
                    print("Switched to analysis mode")
                elif key == ord('s'):
                    self.current_mode = 'suggestion'
                    print("Switched to suggestion mode")
                elif key == ord('t'):
                    self.train_model()
        
        except Exception as e:
            print(f"Error in main loop: {e}")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

# Run the system
if __name__ == "__main__":
    try:
        system = AIAccessorySuggestor()
        system.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a camera connected and OpenCV installed.")