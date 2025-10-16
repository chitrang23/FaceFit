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
    
    # Safety check for zero-length vectors
    if norm_product == 0:
        return 0.0
        
    # Clamp value to prevent arccos errors from floating point inconsistencies
    angle_rad = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
    return np.degrees(angle_rad)

def landmarks_to_feature_vector(landmarks):
    """
    Converts 68 face landmarks into a comprehensive, normalized, fixed-length 
    feature vector for machine learning classification.
    
    Normalization is done using the inter-ocular distance to ensure invariance 
    to head size and distance from the camera.
    
    Args:
        landmarks (list): A list of 68 (x, y) tuples representing the facial landmarks.
        
    Returns:
        np.array: A 1D numpy array containing the calculated features. Returns an
                  empty array if input is invalid.
    """
    if not landmarks or len(landmarks) != 68:
        return np.array([])
        
    # --- 1. Normalization Factor ---
    # Use inter-ocular distance (distance between the eyes) as a robust
    # normalization factor to make features scale-invariant.
    left_eye_center = np.mean(landmarks[36:42], axis=0)
    right_eye_center = np.mean(landmarks[42:48], axis=0)
    
    norm_factor = euclidean_distance(left_eye_center, right_eye_center)
    if norm_factor == 0:
        return np.array([]) # Avoid division by zero
        
    features = []

    # --- 2. Distance & Proportional Features (Normalized) ---
    
    # Overall face dimensions
    face_length = euclidean_distance(landmarks[8], landmarks[27]) / norm_factor
    cheekbone_width = euclidean_distance(landmarks[1], landmarks[15]) / norm_factor
    jaw_width = euclidean_distance(landmarks[4], landmarks[12]) / norm_factor
    forehead_width = euclidean_distance(landmarks[17], landmarks[26]) / norm_factor
    
    # Vertical proportions
    midface_height = euclidean_distance(landmarks[27], landmarks[33]) / norm_factor
    lower_face_height = euclidean_distance(landmarks[33], landmarks[8]) / norm_factor
    
    features.extend([
        face_length,
        cheekbone_width,
        jaw_width,
        forehead_width,
        midface_height,
        lower_face_height
    ])

    # --- 3. Ratio Features ---
    # These features describe the relative proportions of the face, which are
    # key indicators of shape.
    
    # Face Shape Index (Length vs. Width)
    features.append(face_length / cheekbone_width if cheekbone_width > 0 else 0)
    
    # Jaw to Cheekbone Ratio (Indicates tapering towards chin)
    features.append(jaw_width / cheekbone_width if cheekbone_width > 0 else 0)
    
    # Forehead to Jaw Ratio
    features.append(forehead_width / jaw_width if jaw_width > 0 else 0)

    # Relative Chin Width
    chin_width = euclidean_distance(landmarks[6], landmarks[10]) / norm_factor
    features.append(chin_width / jaw_width if jaw_width > 0 else 0)

    # --- 4. Angle Features (Shape Specific) ---
    # Angles provide crucial information about the contours of the face.
    
    # Jaw Angle (Gonions): Distinguishes between "Round" and "Square" shapes.
    # A smaller angle (~90 degrees) indicates a square jaw.
    left_jaw_angle = calculate_angle(landmarks[2], landmarks[4], landmarks[6])
    right_jaw_angle = calculate_angle(landmarks[14], landmarks[12], landmarks[10])
    avg_jaw_angle = (left_jaw_angle + right_jaw_angle) / 2.0
    features.append(avg_jaw_angle)
    
    # Chin Angle (Pointiness): Distinguishes "Heart" or "Long" shapes.
    # A smaller angle indicates a more pointed chin.
    chin_angle = calculate_angle(landmarks[6], landmarks[8], landmarks[10])
    features.append(chin_angle)
    
    # Cheek Angle: Describes the prominence of the cheekbones.
    cheek_angle = calculate_angle(landmarks[2], landmarks[29], landmarks[14])
    features.append(cheek_angle)
    
    return np.array(features)