import numpy as np
from scipy.spatial import distance as dist

# --- Core Helper Functions ---

def euclidean_distance(p1, p2):
    """Calculates the Euclidean distance between two landmark points."""
    return dist.euclidean(p1, p2)

def calculate_angle(p1, p_center, p3):
    """Calculates the angle between three points (p1-p_center-p3) in degrees."""
    v1 = np.array(p1) - np.array(p_center)
    v2 = np.array(p3) - np.array(p_center)
    
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    if norm_product == 0:
        return 0
        
    # Clamp value to prevent arccos errors on floating point inconsistencies
    angle_rad = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
    return np.degrees(angle_rad)

def landmarks_to_feature_vector(landmarks):
    """
    Converts 68 face landmarks into a normalized, fixed-length feature vector 
    for machine learning classification.
    
    Normalization is done using the inter-ocular distance to ensure invariance 
    to head size and distance from camera.
    
    Args:
        landmarks (list): A list of 68 (x, y) tuples from dlib.

    Returns:
        np.array: A 1D feature vector of normalized distances, ratios, and angles.
    """
    if not landmarks or len(landmarks) < 68:
        return np.array([])

    # 1. Normalization Factor: Inter-ocular distance (between eye centers)
    
    # Left eye center (avg of 37-42)
    left_eye_center = np.mean(landmarks[36:42], axis=0) 
    # Right eye center (avg of 43-48)
    right_eye_center = np.mean(landmarks[42:48], axis=0) 
    
    norm_factor = euclidean_distance(left_eye_center, right_eye_center)
    if norm_factor == 0:
        return np.array([])
        
    features = []

    # --- 2. Distance Features (Normalized) ---
    
    # Jaw Width (3 to 13)
    jaw_width = euclidean_distance(landmarks[3], landmarks[13]) / norm_factor
    features.append(jaw_width)
    
    # Face Length (Chin 8 to Center Forehead 27)
    face_length = euclidean_distance(landmarks[8], landmarks[27]) / norm_factor
    features.append(face_length)

    # Forehead Width (17 to 26 - approximate widest forehead points)
    forehead_width = euclidean_distance(landmarks[17], landmarks[26]) / norm_factor
    features.append(forehead_width)

    # Cheekbone Width (1 to 15 - widest points of the face)
    cheekbone_width = euclidean_distance(landmarks[1], landmarks[15]) / norm_factor
    features.append(cheekbone_width)

    # --- 3. Ratio Features ---
    features.append(face_length / jaw_width)
    features.append(jaw_width / forehead_width)
    
    # --- 4. Angle Features (Shape Specific) ---
    
    # Lower jaw angle (to measure roundness/squareness)
    jaw_angle = calculate_angle(landmarks[5], landmarks[8], landmarks[11])
    features.append(jaw_angle)
    
    # Chin Pointedness Angle 
    chin_angle = calculate_angle(landmarks[3], landmarks[8], landmarks[13])
    features.append(chin_angle)
    
    return np.array(features)
