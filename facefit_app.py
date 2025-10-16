import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import dlib
import numpy as np
import pickle
import os
from PIL import Image, ImageTk
import math
import threading # For running AI calls without freezing the UI
import time # NEW: For generating unique filenames

# NEW: Import the Google GenAI SDK
try:
    import google.genai as genai
    GEMINI_CLIENT = None
except ImportError:
    print("WARNING: google-genai not installed. AI suggestions will use fallback data.")
    GEMINI_CLIENT = "FALLBACK" # Sentinel value for static fallback


# --- 1. COLOR PALETTE DEFINITION ---
COLORS = {
    'Black': '#000000',
    'White': '#ffffff',
    'Deep_Indigo': '#31186a',  # Primary Accent / Brand
    'Crimson_Red': '#d9023f',  # Danger / Stop
    'Light_Pink': '#fa8caa',   # Subtle Background / Hover
    'Bright_Violet': '#e78deb', # Input Focus
    'Aqua_Green': '#6af7c1',   # Success / Start
    'Dark_Olive': '#556b2f',   # Subtle Detail / Neutral Contrast
}

# --- Configuration ---
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
CLASSIFIER_MODEL_PATH = "face_shape_classifier.pkl"
ACCESORY_DIR = "accessories"

# --- GLOBAL VARIABLES ---
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
CAP = None
LIVE_STREAMING = False
LAST_FACE_SHAPE = "Unknown"

# Configurable White Removal Thresholds
WHITE_THRESHOLDS = {'glasses': 230, 'hat': 245, 'earrings': 240, 'necklace': 240, 'default': 240}


# --- ACCESSORY GROUPING AND PLACEMENT LOGIC ---
ACCESSORY_KEYWORDS = {
    'Glasses / Sunglasses': ['glass', 'sun', 'spectacle', 'eye', 'frame'],
    'Hats / Headwear': ['cap', 'hat', 'fedora', 'beanie', 'headband', 'crown', 'brim'],
    'Earrings / Jewelry': ['earring', 'hoop', 'stud', 'jewel', 'pendant', 'drop'],
    'Necklaces / Pendants': ['necklace', 'chain', 'pendant', 'scarf', 'tie', 'choker'],
    'Others': []
}

def determine_placement(accessory_name):
    name = accessory_name.lower()
    if any(keyword in name for keyword in ACCESSORY_KEYWORDS['Hats / Headwear']):
        return 'hat'
    if any(keyword in name for keyword in ACCESSORY_KEYWORDS['Glasses / Sunglasses']):
        return 'glasses'
    if any(keyword in name for keyword in ACCESSORY_KEYWORDS['Earrings / Jewelry']):
        return 'earrings'
    if any(keyword in name for keyword in ACCESSORY_KEYWORDS['Necklaces / Pendants']):
        return 'necklace'
    return 'default'

ACCESSORY_CATEGORIES = {}

def organize_accessories():
    global ACCESSORY_CATEGORIES
    ACCESSORY_CATEGORIES = {cat: [] for cat in ACCESSORY_KEYWORDS.keys()}

    if os.path.exists(ACCESORY_DIR):
        accessory_files = sorted([f for f in os.listdir(ACCESORY_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        for filename in accessory_files:
            assigned = False
            for category, keywords in ACCESSORY_KEYWORDS.items():
                if category == 'Others': continue
                if any(keyword in filename.lower() for keyword in keywords):
                    ACCESSORY_CATEGORIES[category].append(filename)
                    assigned = True
                    break
            if not assigned:
                ACCESSORY_CATEGORIES['Others'].append(filename)

    ACCESSORY_CATEGORIES = {k: v for k, v in ACCESSORY_CATEGORIES.items() if v}

organize_accessories()


# --- UTILITY FUNCTIONS (Core Logic) ---

def hex_to_bgr(hex_color):
    """Converts a standard HEX color string to an OpenCV BGR tuple."""
    return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))[::-1]

def _clamp_position(x, y, frame_w, frame_h, acc_w, acc_h):
    """Clamps the top-left corner (x, y) of an accessory to keep it fully inside the frame."""
    x = max(0, min(x, frame_w - acc_w))
    y = max(0, min(y, frame_h - acc_h))
    return x, y

def process_accessory_image(img_data, placement='default'):
    if img_data is None: return None
    accessory_img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_UNCHANGED)
    if accessory_img is None: return None

    if accessory_img.ndim < 3 or accessory_img.shape[2] == 1:
        bgra_img = cv2.cvtColor(accessory_img, cv2.COLOR_GRAY2BGRA)
    elif accessory_img.shape[2] == 3:
        b, g, r = cv2.split(accessory_img)
        alpha = np.full(b.shape, 255, dtype=b.dtype)
        bgra_img = cv2.merge([b, g, r, alpha])
    elif accessory_img.shape[2] == 4:
        bgra_img = accessory_img.copy()
    else:
        return None

    # Use configurable threshold based on placement
    threshold = WHITE_THRESHOLDS.get(placement, WHITE_THRESHOLDS['default'])
    b, g, r, _ = cv2.split(bgra_img)
    mask_white = (b > threshold) & (g > threshold) & (r > threshold)
    bgra_img[:, :, 3][mask_white] = 0
    return bgra_img

def landmarks_to_feature_vector(landmarks):
    if not landmarks or len(landmarks) < 68: return np.array([])
    landmarks_np = np.array(landmarks)
    try:
        jaw_width = np.linalg.norm(landmarks_np[3] - landmarks_np[13])
        forehead_width = np.linalg.norm(landmarks_np[17] - landmarks_np[26])
        face_length = np.linalg.norm(landmarks_np[8] - landmarks_np[27])
    except IndexError: return np.array([])
    ratio_length_jaw = face_length / jaw_width if jaw_width > 0 else 0
    ratio_jaw_forehead = jaw_width / forehead_width if forehead_width > 0 else 0
    return np.array([ratio_length_jaw, ratio_jaw_forehead])

def detect_and_get_landmarks(image_cv2):
    if PREDICTOR is None: return None, None
    img_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    faces = DETECTOR(img_rgb, 1)
    if not faces: return None, None
    face = faces[0]
    landmarks = PREDICTOR(img_rgb, face)
    points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 68)]
    return face, points

def rotate_accessory(accessory_img, angle_deg, target_width, target_height):
    resized_acc = cv2.resize(accessory_img, (target_width, target_height), interpolation=cv2.INTER_AREA)

    h_acc, w_acc, _ = resized_acc.shape
    M = cv2.getRotationMatrix2D((w_acc / 2, h_acc / 2), angle_deg, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    new_w = int((h_acc * sin) + (w_acc * cos))
    new_h = int((h_acc * cos) + (w_acc * sin))

    M[0, 2] += (new_w / 2) - (w_acc / 2)
    M[1, 2] += (new_h / 2) - (h_acc / 2)

    rotated_acc = cv2.warpAffine(resized_acc, M, (new_w, new_h),
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0, 0))

    return rotated_acc, new_w, new_h

def _apply_single_overlay(frame, accessory_img_resized, x_offset, y_offset):
    h_acc, w_acc, _ = accessory_img_resized.shape
    frame_h, frame_w = frame.shape[0], frame.shape[1]

    # --- Clamp position to ensure accessory stays within the frame ---
    x_offset, y_offset = _clamp_position(x_offset, y_offset, frame_w, frame_h, w_acc, h_acc)
    # -----------------------------------------------------------------

    # Calculate overlay coordinates based on clamped offsets
    y1, y2 = y_offset, y_offset + h_acc
    x1, x2 = x_offset, x_offset + w_acc

    # Recalculate frame/accessory slices
    y1_frame, x1_frame = max(0, y1), max(0, x1)
    y2_frame, x2_frame = min(frame_h, y2), min(frame_w, x2)

    y1_acc = y1_frame - y1
    y2_acc = y2_frame - y1
    x1_acc = x1_frame - x1
    x2_acc = x2_frame - x1

    h_slice = y2_frame - y1_frame
    w_slice = x2_frame - x1_frame

    if h_slice <= 0 or w_slice <= 0:
        return frame

    acc_slice = accessory_img_resized[y1_acc:y2_acc, x1_acc:x2_acc]
    roi = frame[y1_frame:y2_frame, x1_frame:x2_frame]

    bgr_acc = acc_slice[:, :, :3]
    alpha_acc = acc_slice[:, :, 3]

    alpha_mask = alpha_acc.astype(np.float32) / 255.0
    alpha_mask_3ch = cv2.cvtColor(alpha_mask, cv2.COLOR_GRAY2BGR)

    blended_roi = (bgr_acc.astype(np.float32) * alpha_mask_3ch) + \
                  (roi.astype(np.float32) * (1.0 - alpha_mask_3ch))

    frame[y1_frame:y2_frame, x1_frame:x2_frame] = blended_roi.astype(np.uint8)
    return frame

# MODIFIED: Includes rotation_offset handling
def overlay_accessory(frame, accessory_img, landmarks, placement='default', overrides=None):
    if PREDICTOR is None or landmarks is None or len(landmarks) < 68 or accessory_img.shape[2] != 4:
        return frame

    landmarks_np = np.array(landmarks)

    # --- 1. Calculate Face Roll Angle ---
    p_left_eye_outer = landmarks_np[36]
    p_right_eye_outer = landmarks_np[45]
    dx = p_right_eye_outer[0] - p_left_eye_outer[0]
    dy = p_right_eye_outer[1] - p_left_eye_outer[1]
    angle_deg = np.degrees(np.arctan2(dy, dx))

    # --- APPLY USER ROTATION OVERRIDE ---
    rotation_offset = overrides.get('rotation_offset', 0) if overrides else 0
    final_rotation_angle = angle_deg + rotation_offset

    # --- 2. Calculate Scaling and Center Points ---
    p_left_temple = landmarks_np[0]
    p_right_temple = landmarks_np[16]
    p_nose_bridge = landmarks_np[27]
    p_chin = landmarks_np[8]
    p_left_eye_inner = landmarks_np[39]
    p_right_eye_inner = landmarks_np[42]

    face_width = np.linalg.norm(p_right_temple - p_left_temple)
    center_x_face = (p_left_temple[0] + p_right_temple[0]) // 2

    # --- APPLY USER SCALE OVERRIDE ---
    scale_factor = overrides.get('scale_factor', 1.0) if overrides else 1.0

    target_width_multiplier = 1.3
    if placement == 'hat':
        target_width_multiplier = 1.5
    elif placement == 'earrings':
        target_width_multiplier = 0.4
    elif placement == 'necklace':
        target_width_multiplier = 1.6

    h_acc_orig, w_acc_orig, _ = accessory_img.shape

    if placement == 'glasses' or placement == 'default':
        eye_distance = np.linalg.norm(landmarks_np[45] - landmarks_np[36])
        target_width = int(eye_distance * 1.8 * scale_factor) # Apply scale_factor
    else:
        target_width = int(face_width * target_width_multiplier * scale_factor) # Apply scale_factor

    target_height = int((target_width / w_acc_orig) * h_acc_orig)

    if target_width < 10 or target_height < 10: return frame

    # --- 3. Rotate and Resize the Accessory (Using FINAL ANGLE) ---
    rotated_acc, new_w, new_h = rotate_accessory(accessory_img, final_rotation_angle, target_width, target_height)

    # --- 4. Positioning Logic (x_offset, y_offset are the top-left corner) ---

    # Helper to retrieve the override safely
    # If the user hasn't moved the accessory, the offset will be None (or 1.0 for scale).
    # We apply the user's manual offset relative to the calculated default position.
    def get_override_pos_delta(key, default_val):
        user_delta = overrides.get(key, 0) if overrides and overrides.get(key) is not None else 0
        return default_val + user_delta


    if placement == 'glasses' or placement == 'default':
        center_x_target = (p_left_eye_inner[0] + p_right_eye_inner[0]) // 2
        center_y_eye_line = (landmarks_np[36][1] + landmarks_np[45][1]) // 2
        center_y_target = center_y_eye_line + int(target_height * 0.10)

        default_x_offset = center_x_target - int(new_w / 2)
        default_y_offset = center_y_target - int(new_h / 2)

        # Apply override delta
        x_offset = get_override_pos_delta('x_offset', default_x_offset)
        y_offset = get_override_pos_delta('y_offset', default_y_offset)

    elif placement == 'hat':
        center_y_face = p_nose_bridge[1] - int(face_width * 0.5)

        default_x_offset = center_x_face - int(new_w / 2)
        default_y_offset = center_y_face - int(new_h / 2)

        # Apply override delta
        x_offset = get_override_pos_delta('x_offset', default_x_offset)
        y_offset = get_override_pos_delta('y_offset', default_y_offset)

    elif placement == 'earrings':
        # Left Earring Default Position
        earring_center_x_left = p_left_temple[0]
        earring_center_y_left = p_left_temple[1] + int(face_width * 0.05)

        default_x_offset_left = earring_center_x_left - int(new_w / 2)
        default_y_offset_left = earring_center_y_left - int(new_h / 2)

        # Apply override delta for left earring
        user_dx_delta = overrides.get('x_offset', 0) if overrides and overrides.get('x_offset') is not None else 0
        user_dy_delta = overrides.get('y_offset', 0) if overrides and overrides.get('y_offset') is not None else 0

        x_offset_left = default_x_offset_left + user_dx_delta
        y_offset_left = default_y_offset_left + user_dy_delta

        frame = _apply_single_overlay(frame, rotated_acc, x_offset_left, y_offset_left)

        # Right Earring Default Position
        earring_center_x_right = p_right_temple[0]
        earring_center_y_right = p_right_temple[1] + int(face_width * 0.05)

        default_x_offset_right = earring_center_x_right - int(new_w / 2)
        default_y_offset_right = earring_center_y_right - int(new_h / 2)

        # Apply the same displacement (user_dx_delta/user_dy_delta) to the right earring
        x_offset_right = default_x_offset_right + user_dx_delta
        y_offset_right = default_y_offset_right + user_dy_delta

        return _apply_single_overlay(frame, rotated_acc, x_offset_right, y_offset_right)

    elif placement == 'necklace':
        neck_y_start = p_chin[1]

        default_x_offset = center_x_face - int(new_w / 2)
        default_y_offset = neck_y_start

        # Apply override delta
        x_offset = get_override_pos_delta('x_offset', default_x_offset)
        y_offset = get_override_pos_delta('y_offset', default_y_offset)

    return _apply_single_overlay(frame, rotated_acc, x_offset, y_offset)

def classify_face_shape(landmarks):
    global ML_MODEL_LOADED, FACE_SHAPE_MODEL
    if not landmarks: return "Unknown"

    if ML_MODEL_LOADED and FACE_SHAPE_MODEL is not None:
        try:
            features = landmarks_to_feature_vector(landmarks)
            if features.size > 0:
                return FACE_SHAPE_MODEL.predict([features])[0]
        except Exception:
            pass

    landmarks_np = np.array(landmarks)
    try:
        jaw_width = np.linalg.norm(landmarks_np[3] - landmarks_np[13])
        forehead_width = np.linalg.norm(landmarks_np[17] - landmarks_np[26])
        face_length = np.linalg.norm(landmarks_np[8] - landmarks_np[27])
    except IndexError:
        return "Unknown"

    ratio_length_jaw = face_length / jaw_width if jaw_width > 0 else 0
    ratio_jaw_forehead = jaw_width / forehead_width if forehead_width > 0 else 0

    if ratio_length_jaw >= 1.4:
        return "Long"
    elif ratio_length_jaw > 1.25 and ratio_jaw_forehead < 1.0:
        return "Heart"
    elif ratio_length_jaw >= 1.15:
        return "Oval"
    elif ratio_length_jaw >= 1.0 and abs(jaw_width - forehead_width) < (jaw_width * 0.1):
        return "Square"
    else:
        return "Round"

# --- Initialization and Error Handling (Unchanged) ---
DETECTOR = dlib.get_frontal_face_detector()
FACE_SHAPE_MODEL = None
ML_MODEL_LOADED = False
PREDICTOR = None
DLIB_LOAD_ERROR = False

try:
    if os.path.exists(SHAPE_PREDICTOR_PATH):
        PREDICTOR = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    else:
        DLIB_LOAD_ERROR = True
        print(f"ERROR: Dlib shape predictor file not found at: {SHAPE_PREDICTOR_PATH}")

    if os.path.exists(CLASSIFIER_MODEL_PATH):
        with open(CLASSIFIER_MODEL_PATH, 'rb') as f:
            FACE_SHAPE_MODEL = pickle.load(f)
            ML_MODEL_LOADED = True
    else:
        print(f"WARNING: ML model file '{CLASSIFIER_MODEL_PATH}' not found. Using geometric fallback.")

except Exception as e:
    DLIB_LOAD_ERROR = True
    print(f"FATAL ERROR during Dlib/Model Load: {e}")
    PREDICTOR = None

# --- ACCESSORY SUGGESTIONS (STATIC FALLBACK) ---
ACCESSORIES_DB = {
    'Round': {
        'Optical Frames': {'text': 'Frames that are wider than they are tall (Rectangle, Square, Geometric). Aim for dark colors or patterns to add definition. A high, clear bridge helps elongate the nose.'},
        'Sun & Protective Wear': {'text': 'Angular styles like Square, Wayfarer, or Cat-Eye to create contrast. Avoid small, round, or rimless styles. Look for thick, prominent temples to add width.'},
        'Hats & Headwear': {'text': 'Hats with tall crowns and straight brims (e.g., Fedora, Trilby). Baseball caps with structured, high fronts. Avoid round-crowned or floppy hats.'},
        'Earrings & Studs': {'text': 'Long, dangling, or angular drop earrings (Rectangle, Oval, Triangle shapes). These draw the eye vertically, slimming the face. Avoid small studs or chunky hoops.'},
        'Necklaces & Pendants': {'text': 'Long necklaces (20+ inches) with V-shaped or Y-shaped pendants. These create a vertical line, visually lengthening the neck and face.'},
    },
    'Oval': {
        'Optical Frames': {'text': 'You suit most styles! Medium-to-large frames, such as Walnut or Aviator shapes, that are as wide as the broadest part of the face. Avoid frames that are too narrow or too wide.'},
        'Sun & Protective Wear': {'text': 'Oversized, vintage styles, or bold Aviators. Experiment with unique colors and textures. Butterfly and shield sunglasses also work well.'},
        'Hats & Headwear': {'text': 'Highly versatile. Wide-brimmed hats, Felt Fedoras, or simple Beanies. Ensure the brim is no wider than your shoulders for balance.'},
        'Earrings & Studs': {'text': 'Studs, medium-sized Hoops, and subtle Teardrops. You can wear almost any style, but simple designs showcase your natural symmetry.'},
        'Necklaces & Pendants': {'text': 'Short chains (Chokers, 14-16 inches) or medium lengths (18-20 inches) to highlight the collarbone. Layering necklaces is an excellent choice.'},
    },
    'Square': {
        'Optical Frames': {'text': 'Round, Oval, or Cat-Eye frames (to soften the strong jawline). Choose thinner metal or plastic frames, and keep the bridge low to decrease face length.'},
        'Sun & Protective Wear': {'text': 'Curvy frames like Round, Butterfly, or Cat-Eye. Choose light colors or gradient lenses. Avoid square or geometric shapes, which can emphasize angularity.'},
        'Hats & Headwear': {'text': 'Round crowns and floppy brims (e.g., Cloche, Beret, Floppy Sun Hat) to soften lines. Wear hats slightly tilted for an asymmetrical look.'},
        'Earrings & Studs': {'text': 'Round Hoops, Studs, and medium-length, softly curved drops. The key is to add curve and avoid sharp angles near the jawline.'},
        'Necklaces & Pendants': {'text': 'Long necklaces with large, curved pendants (e.g., Circle, Round, or Abstract shapes). Length helps elongate the chin.'},
    },
    'Heart': {
        'Optical Frames': {'text': 'Bottom-heavy frames (like Modified Aviators), Rimless, or Cat-Eye (with a narrow base) to balance the narrow chin. Light colors or frames with detailing on the bottom half are ideal.'},
        'Sun & Protective Wear': {'text': 'Aviators, rimless styles, or semi-rimless frames. Look for round or slightly curved lens shapes. Avoid heavy tops or strong vertical lines.'},
        'Hats & Headwear': {'text': 'Medium brims, Bucket hats, or Cloches. Wear them pushed forward to minimize the forehead. Avoid small, fitted hats like fedoras.'},
        'Earrings & Studs': {'text': 'Teardrop or Chandelier earrings that are narrow at the top and wide at the bottom. This adds volume near the jaw to balance the forehead.'},
        'Necklaces & Pendants': {'text': 'Short chains or Chokers (14-16 inches). These draw attention upwards and soften the chin line.'},
    },
    'Long': {
        'Optical Frames': {'text': 'Deep frames with thick, decorative temples (Square, Tall Rectangle) to add width and break up the length. Avoid small or narrow frames. Dark, contrasting bridges help reduce length.'},
        'Sun & Protective Wear': {'text': 'Tall, large, or geometric frames. Look for exaggerated details on the sides/temples. Wayfarers and oversized frames are excellent choices.'},
        'Hats & Headwear': {'text': 'Low crowns and wide brims (e.g., Sun hats, Cowboy hats, Newsboy cap). This reduces the appearance of height. Avoid very tall, structured hats.'},
        'Earrings & Studs': {'text': 'Volume-adding, wide, or statement earrings (large Studs, wide Hoops). Avoid long, vertical drops as they emphasize length.'},
        'Necklaces & Pendants': {'text': 'Short, chunky necklaces or Chokers (14-16 inches). The horizontal line created adds width and breaks up the face length.'},
    },
    'Unknown': {
        'Optical Frames': {'text': 'Try a neutral, medium size frame (e.g., light brown rectangular).'},
        'Sun & Protective Wear': {'text': 'Try standard Aviators or classic Wayfarers.'},
        'Hats & Headwear': {'text': 'A simple beanie or ball cap.'},
        'Earrings & Studs': {'text': 'Simple studs or medium hoops.'},
        'Necklaces & Pendants': {'text': 'A simple, medium-length chain (18 inches).'},
    }
}


# --- AI SUGGESTION LOGIC (Unchanged) ---

def configure_gemini():
    """Confgures the Gemini client using the environment variable."""
    global GEMINI_CLIENT
    if GEMINI_CLIENT == "FALLBACK":
        return

    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            # We use an explicit API key configuration if available
            GEMINI_CLIENT = genai.Client(api_key=api_key)
            print("Gemini Client successfully configured.")
        except Exception as e:
            GEMINI_CLIENT = "FALLBACK"
            print(f"ERROR configuring Gemini client: {e}. Falling back to static suggestions.")
    else:
        GEMINI_CLIENT = "FALLBACK"
        print("WARNING: GEMINI_API_KEY not found. Falling back to static suggestions.")


def generate_ai_suggestion(face_shape, gender, accessory_type):
    """Calls the Gemini API to get a dynamic, nuanced suggestion."""
    global GEMINI_CLIENT

    if GEMINI_CLIENT is None or GEMINI_CLIENT == "FALLBACK":
         static_text = ACCESSORIES_DB.get(face_shape, ACCESSORIES_DB['Unknown']).get(accessory_type, {'text': "AI unavailable. Using geometric rules."}).get('text')
         if gender == 'Male' and accessory_type == 'Earrings & Studs':
             return 'N/A (Filtered by Male Profile, unless desired)'
         return static_text

    try:
        system_prompt = (
            "You are a professional, highly creative fashion stylist specializing in face shape analysis. "
            "Your suggestions must be specific (e.g., 'Thin rose gold wire-frame cat-eye glasses' not 'Cat-Eye glasses'). "
            f"Provide trending accessory advice for a {face_shape} face and a {gender} profile. "
            "Keep the explanation concise and highly descriptive, exactly 2-3 sentences long."
        )

        user_prompt = (
            f"Please give me a specific, fashionable suggestion for **{accessory_type}** that suits a **{face_shape}** face."
        )

        response = GEMINI_CLIENT.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_prompt,
            config={'system_instruction': system_prompt}
        )
        return response.text

    except Exception as e:
        print(f"AI API Error for {accessory_type}: {e}. Returning static fallback.")
        static_text = ACCESSORIES_DB.get(face_shape, ACCESSORIES_DB['Unknown']).get(accessory_type, {'text': "AI suggestion failed. Try geometric rules."}).get('text')
        if gender == 'Male' and accessory_type == 'Earrings & Studs':
             return 'N/A (Filtered by Male Profile, unless desired)'
        return static_text

# --- Tkinter Application Class ---
class FaceFitApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("FaceFit AI: Virtual Try-On")
        self.geometry("1000x650")

        configure_gemini()

        self.configure_styles()

        self.gender_var = tk.StringVar(value='Female')
        self.category_var = tk.StringVar(value=list(ACCESSORY_CATEGORIES.keys())[0] if ACCESSORY_CATEGORIES else "")
        self.accessory_files = []
        self.current_face_shape = tk.StringVar(value="N/A")
        self.image_input_cv2 = None
        self.accessory_listbox_var = tk.StringVar()
        self.ai_suggestions = {}
        self.suggestion_thread = None

        # --- PERFORMANCE & TRACKING VARIABLES (OPTIMIZED) ---
        self.last_frame_processed = None
        self.frame_count = 0
        self.process_interval = 5 # Run full detection every 5 frames
        self.last_landmarks = None
        self.face_tracker = None # For dlib's correlation tracker
        self.tracker_active = False # Flag to indicate if the tracker is in use

        # --- Drag/Resize/Rotation Variables ---
        self.selected_accessory_name = None # Name of the accessory currently being edited
        # Stores user overrides: {name: {'x_offset': int, 'y_offset': int, 'scale_factor': float, 'rotation_offset': float}}
        self.accessory_overrides = {}
        self.ROTATION_INCREMENT = 3 # Degrees to rotate per key press

        self.tk_setPalette(background=COLORS['White'])

        main_paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = ttk.Frame(main_paned, width=650, style='Custom.TFrame')
        main_paned.add(left_frame, weight=3)
        self.setup_display_tabs(left_frame)

        right_frame = ttk.Frame(main_paned, width=300, style='Custom.TFrame')
        main_paned.add(right_frame, weight=1)

        self.setup_controls(right_frame)
        self.setup_results(right_frame)

        if DLIB_LOAD_ERROR:
            messagebox.showerror("Initialization Error", "Failed to load dlib files or models. Face detection will be limited.")

        self.update_suggestions()

        # --- Bind keyboard events for resizing, moving, and rotation ---

        # Resizing (Use '+' and '-' for increasing/decreasing scale)
        self.bind('+', lambda event: self.change_accessory_scale(0.05))
        self.bind('-', lambda event: self.change_accessory_scale(-0.05))

        # Fine-Tuning Movement (Arrow Keys)
        self.bind('<Left>', lambda event: self.move_accessory('x', -5))
        self.bind('<Right>', lambda event: self.move_accessory('x', 5))
        self.bind('<Up>', lambda event: self.move_accessory('y', -5))
        self.bind('<Down>', lambda event: self.move_accessory('y', 5))

        # Rotation (Q and E keys)
        self.bind('q', lambda event: self.change_accessory_rotation(-self.ROTATION_INCREMENT)) # Q: Counter-Clockwise
        self.bind('e', lambda event: self.change_accessory_rotation(self.ROTATION_INCREMENT))  # E: Clockwise

        # Resetting (Use Delete or Backspace)
        self.bind('<Delete>', lambda event: self.clear_accessory_override(self.selected_accessory_name))
        self.bind('<BackSpace>', lambda event: self.clear_accessory_override(self.selected_accessory_name))


    # --- STYLE & SETUP METHODS (Unchanged) ---
    def configure_styles(self,):
        s = ttk.Style()
        s.theme_use('default')

        s.configure('Custom.TFrame', background=COLORS['White'])
        s.configure('Custom.TLabel', background=COLORS['White'], foreground=COLORS['Black'], font=('Arial', 10))
        s.configure('Header.TLabel', background=COLORS['White'], foreground=COLORS['Deep_Indigo'], font=('Arial', 10, 'bold'))
        s.configure('Result.TLabel', background=COLORS['Light_Pink'], foreground=COLORS['Deep_Indigo'], font=('Arial', 14, 'bold'), padding=5)

        s.configure('TNotebook', background=COLORS['White'], borderwidth=0)
        s.configure('TNotebook.Tab', background=COLORS['Light_Pink'], foreground=COLORS['Black'], padding=[10, 5])
        s.map('TNotebook.Tab',
              background=[('selected', COLORS['Deep_Indigo']), ('active', COLORS['Light_Pink'])],
              foreground=[('selected', COLORS['White']), ('active', COLORS['Black'])],
              expand=[('selected', [1, 1, 1, 0])])

        s.configure('Primary.TButton', background=COLORS['Deep_Indigo'], foreground=COLORS['White'], font=('Arial', 10, 'bold'), borderwidth=0)
        s.map('Primary.TButton', background=[('active', COLORS['Bright_Violet']), ('!disabled', COLORS['Deep_Indigo'])])

        s.configure('Success.TButton', background=COLORS['Aqua_Green'], foreground=COLORS['Black'], font=('Arial', 10, 'bold'), borderwidth=0)
        s.map('Success.TButton', background=[('active', COLORS['Dark_Olive']), ('!disabled', COLORS['Aqua_Green'])])

        s.configure('Danger.TButton', background=COLORS['Crimson_Red'], foreground=COLORS['White'], font=('Arial', 10, 'bold'), borderwidth=0)
        s.map('Danger.TButton', background=[('active', COLORS['Light_Pink']), ('!disabled', COLORS['Crimson_Red'])])

        s.configure('TLabelFrame', background=COLORS['White'], bordercolor=COLORS['Dark_Olive'])
        s.configure('TLabelFrame.Label', background=COLORS['White'], foreground=COLORS['Deep_Indigo'], font=('Arial', 11, 'bold'))

        s.configure('TCombobox', fieldbackground=COLORS['White'], background=COLORS['White'], foreground=COLORS['Black'])

    def setup_display_tabs(self, parent_frame):
        self.notebook = ttk.Notebook(parent_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.webcam_tab = ttk.Frame(self.notebook, style='Custom.TFrame')
        self.notebook.add(self.webcam_tab, text='📹 Live Webcam')
        self.webcam_display_frame = ttk.Frame(self.webcam_tab, style='Custom.TFrame')
        self.webcam_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.video_label = ttk.Label(self.webcam_display_frame, text="Click 'Start Webcam' to begin live try-on. Use Arrow Keys to move, +/- to resize, and Q/E to rotate selected accessory.", style='Custom.TLabel', anchor='center')
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self.upload_tab = ttk.Frame(self.notebook, style='Custom.TFrame')
        self.notebook.add(self.upload_tab, text='🖼️ Upload Photo')
        self.upload_display_frame = ttk.Frame(self.upload_tab, style='Custom.TFrame')
        self.upload_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Button(self.upload_display_frame, text="1. Select Photo for Try-On", command=self.load_image_file, style='Primary.TButton').pack(fill=tk.X, pady=5)

        self.static_image_label = ttk.Label(self.upload_display_frame, text="Uploaded image will appear here. Use Arrow Keys to move, +/- to resize, and Q/E to rotate selected accessory.", style='Custom.TLabel', anchor='center')
        self.static_image_label.pack(fill=tk.BOTH, expand=True)

        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

    def on_tab_change(self, event):
        selected_tab = self.notebook.tab(self.notebook.select(), "text")
        if selected_tab != '📹 Live Webcam':
            self.stop_webcam()

    def setup_controls(self, parent_frame):
        profile_group = ttk.LabelFrame(parent_frame, text="User Profile", padding="10 10 10 10")
        profile_group.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(profile_group, text="Gender:", style='Custom.TLabel').grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.gender_dropdown = ttk.Combobox(profile_group, textvariable=self.gender_var,
                      values=['Female', 'Male', 'Other', 'Prefer not to say'], state='readonly')
        self.gender_dropdown.grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        self.gender_dropdown.bind('<<ComboboxSelected>>', lambda e: self.update_suggestions())

        accessory_group = ttk.LabelFrame(parent_frame, text="Select Accessories (Multi-Select)", padding="10 10 10 10")
        accessory_group.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(accessory_group, text="Category:", style='Custom.TLabel').pack(fill=tk.X, padx=5, pady=2)

        self.category_dropdown = ttk.Combobox(accessory_group,
                                              textvariable=self.category_var,
                                              values=list(ACCESSORY_CATEGORIES.keys()),
                                              state='readonly')
        self.category_dropdown.pack(fill=tk.X, padx=5, pady=2)
        self.category_dropdown.bind('<<ComboboxSelected>>', self.update_accessory_listbox)

        ttk.Label(accessory_group, text="Files (Ctrl/Shift):", style='Custom.TLabel').pack(fill=tk.X, padx=5, pady=2)

        self.accessory_listbox = tk.Listbox(accessory_group,
                                             height=6,
                                             selectmode=tk.MULTIPLE,
                                             listvariable=self.accessory_listbox_var,
                                             bg=COLORS['White'],
                                             fg=COLORS['Black'],
                                             selectbackground=COLORS['Bright_Violet'],
                                             selectforeground=COLORS['White'],
                                             borderwidth=1,
                                             relief='flat')
        self.accessory_listbox.pack(fill=tk.X, padx=5, pady=2)

        self.accessory_listbox.bind('<<ListboxSelect>>', self.load_selected_accessories)
        self.update_accessory_listbox()

        webcam_group = ttk.LabelFrame(parent_frame, text="Camera Control", padding="10 10 10 10")
        webcam_group.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(webcam_group, text="Start Webcam", command=self.start_webcam, style='Success.TButton').grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        ttk.Button(webcam_group, text="Stop Webcam", command=self.stop_webcam, style='Danger.TButton').grid(row=0, column=1, sticky='ew', padx=5, pady=5)

        # --- NEW: Save Favorite Button ---
        ttk.Button(webcam_group, text="💾 Save Favorite Look",
                   command=self.save_favorite_look,
                   style='Primary.TButton').grid(row=1, column=0, columnspan=2, sticky='ew', padx=5, pady=5)


    def update_accessory_listbox(self, event=None):
        """Updates the listbox with accessories for the selected category."""
        category = self.category_var.get()
        files = ACCESSORY_CATEGORIES.get(category, [])

        self.accessory_listbox.selection_clear(0, tk.END)
        self.accessory_listbox.delete(0, tk.END)

        for f in files:
            self.accessory_listbox.insert(tk.END, f)

        self.accessory_files = files
        self.load_selected_accessories()

    # --- WEBCAM/PROCESSING METHODS (OPTIMIZED with TRACKER) ---
    def process_webcam_frame(self):
        global CAP, LIVE_STREAMING, LAST_FACE_SHAPE

        if not LIVE_STREAMING or CAP is None:
            self.stop_webcam()
            return

        ret, frame = CAP.read()
        if not ret:
            # If no frame, try again shortly
            self.after(20, self.process_webcam_frame)
            return

        # Resize once at the beginning
        processed_frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        self.last_frame_processed = processed_frame # Store for saving

        # --- ENHANCED TRACKING & THROTTLING LOGIC ---
        run_full_detection = (self.frame_count % self.process_interval == 0)
        self.frame_count += 1

        current_landmarks = None

        # If the tracker is running and it's not time for a full detection, use the tracker.
        if self.tracker_active and not run_full_detection:
            # FAST: Update the tracker and get the new face position
            self.face_tracker.update(processed_frame)
            pos = self.face_tracker.get_position()
            face_rect = dlib.rectangle(int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom()))

            # Run the (relatively fast) landmark prediction on the tracked region
            if PREDICTOR:
                landmarks_dlib = PREDICTOR(processed_frame, face_rect)
                current_landmarks = [(landmarks_dlib.part(i).x, landmarks_dlib.part(i).y) for i in range(0, 68)]
                self.last_landmarks = current_landmarks # Cache the tracked landmarks
        else:
            # EXPENSIVE: Run full face detection
            self.tracker_active = False # Assume tracker is lost until a face is found
            face, landmarks = detect_and_get_landmarks(processed_frame)

            if landmarks and len(landmarks) == 68:
                current_landmarks = landmarks
                self.last_landmarks = landmarks

                # A face was found! Initialize the tracker for subsequent frames.
                self.face_tracker = dlib.correlation_tracker()
                self.face_tracker.start_track(processed_frame, face) # 'face' is already a dlib.rectangle
                self.tracker_active = True

                # Update UI elements only on fresh detection to save resources
                new_shape = classify_face_shape(landmarks)
                if LAST_FACE_SHAPE != new_shape:
                    LAST_FACE_SHAPE = new_shape
                    self.current_face_shape.set(LAST_FACE_SHAPE)
                    self.update_suggestions()
            else:
                # No face found, so ensure tracker is off
                self.current_face_shape.set("Face Not Found")
                self.last_landmarks = None
                self.tracker_active = False

        # --- RENDERING (This part remains the same but gets smoother data) ---
        if current_landmarks:
            landmarks_np = np.array(current_landmarks)
            x, y, w, h = cv2.boundingRect(landmarks_np)

            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), hex_to_bgr(COLORS['Dark_Olive']), 2)

            for name, img_data in self.selected_accessories.items():
                placement = determine_placement(name)
                overrides = self.accessory_overrides.get(name)
                processed_frame = overlay_accessory(processed_frame, img_data, current_landmarks, placement, overrides)

        # --- DISPLAY FRAME ---
        img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Re-queue the next frame processing. A shorter delay (20ms ≈ 50fps) will feel much smoother.
        self.after(20, self.process_webcam_frame)


    # --- AI/SUGGESTION METHODS (Unchanged) ---

    def get_accessory_suggestion_async(self):
        """Worker function to call the AI API in a separate thread."""
        face_shape = self.current_face_shape.get()
        gender = self.gender_var.get()

        accessory_types = [
            'Optical Frames', 'Sun & Protective Wear', 'Hats / Headwear',
            'Earrings / Studs', 'Necklaces / Pendants'
        ]

        new_suggestions = {}
        for acc_type in accessory_types:
            suggestion_text = generate_ai_suggestion(face_shape, gender, acc_type)
            new_suggestions[acc_type] = {'text': suggestion_text}

        self.after(0, lambda: self.display_suggestions(new_suggestions))

    def update_suggestions(self):
        """Initiates the AI suggestion generation thread."""

        for widget in self.suggestions_frame.winfo_children():
            widget.destroy()

        if self.current_face_shape.get() not in ["N/A", "Face Not Found"]:
            loading_message = "🧠 Generating dynamic suggestions (may take a moment)..."
        else:
            loading_message = "Face shape needed for dynamic suggestions."

        loading_label = ttk.Label(self.suggestions_frame, text=loading_message, style='Custom.TLabel', foreground=COLORS['Deep_Indigo'], wraplength=270)
        loading_label.pack(fill=tk.X, padx=5, pady=10)

        if self.current_face_shape.get() not in ["N/A", "Face Not Found"]:
            if self.suggestion_thread is None or not self.suggestion_thread.is_alive():
                self.suggestion_thread = threading.Thread(target=self.get_accessory_suggestion_async)
                self.suggestion_thread.daemon = True
                self.suggestion_thread.start()

    def setup_results(self, parent_frame):
        results_group = ttk.LabelFrame(parent_frame, text="Results & Suggestions", padding="10 10 10 10")
        results_group.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Label(results_group, text="Detected Face Shape:", style='Header.TLabel').pack(fill=tk.X, padx=5, pady=(0, 2))
        ttk.Label(results_group, textvariable=self.current_face_shape, style='Result.TLabel').pack(fill=tk.X, padx=5, pady=(0, 5))

        ttk.Label(results_group, text="Accessory Fit Advice:", style='Header.TLabel').pack(fill=tk.X, padx=5, pady=(5, 2))

        self.suggestions_frame = ttk.Frame(results_group, style='Custom.TFrame')
        self.suggestions_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.display_suggestions(self.ai_suggestions)

    def display_suggestions(self, suggestions):
        """Displays the results of the (AI or Fallback) suggestions in the GUI."""
        self.ai_suggestions = suggestions

        for widget in self.suggestions_frame.winfo_children():
            widget.destroy()

        canvas = tk.Canvas(self.suggestions_frame, bg=COLORS['White'], highlightthickness=0)
        v_scrollbar = ttk.Scrollbar(self.suggestions_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=COLORS['White'])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=v_scrollbar.set)

        v_scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        bg_colors = [COLORS['White'], '#fce4f0']
        row = 0

        if not suggestions:
             tk.Label(scrollable_frame, text="Click 'Start Webcam' or upload a photo to get personalized advice.",
                      wraplength=270, justify=tk.LEFT, background=COLORS['White'], foreground=COLORS['Dark_Olive']).grid(row=0, column=0, sticky='w', padx=5, pady=5)
             return

        for i, (title, item) in enumerate(suggestions.items()):
            bg = bg_colors[i % 2]

            tk.Label(scrollable_frame, text=f"• {title}:", font=('Arial', 10, 'bold'),
                      background=bg,
                      foreground=COLORS['Deep_Indigo'], anchor='nw', justify=tk.LEFT).grid(row=row, column=0, sticky='nw', padx=5, pady=(5,0), columnspan=2)
            row += 1

            tk.Label(scrollable_frame, text=item.get('text','N/A'), wraplength=270,
                      justify=tk.LEFT, font=('Arial', 10),
                      background=bg,
                      foreground=COLORS['Black'], anchor='w').grid(row=row, column=0, sticky='w', padx=15, pady=(0,5), columnspan=2)
            row += 1

    # --- KEYBOARD CONTROL METHODS (UPDATED) ---

    def move_accessory(self, axis, delta):
        """Moves the selected accessory by 'delta' along the specified 'axis' (x or y)."""
        if not self.selected_accessory_name:
            return

        override = self.accessory_overrides[self.selected_accessory_name]
        key = f'{axis}_offset'

        # If offset is None, initialize it to 0 (meaning no displacement from default)
        if override.get(key) is None:
            override[key] = 0

        # Apply the movement delta
        override[key] += delta

        print(f"Moved {self.selected_accessory_name} {axis.upper()} by {delta}. New displacement: {override[key]}")

        # Trigger redraw
        if self.notebook.tab(self.notebook.select(), "text") == '🖼️ Upload Photo':
            self.display_static_image(self.image_input_cv2.copy())
        # Live webcam handles redraw automatically.

    def change_accessory_scale(self, delta):
        """Increases or decreases the scale factor of the selected accessory."""
        if not self.selected_accessory_name:
            return

        override = self.accessory_overrides[self.selected_accessory_name]

        current_scale = override.get('scale_factor', 1.0)

        # Ensure scale factor stays within a reasonable range (0.1 to 3.0)
        new_scale = max(0.1, min(3.0, current_scale + delta))

        override['scale_factor'] = new_scale

        print(f"Changed {self.selected_accessory_name} scale to: {new_scale:.2f}")

        # Trigger redraw
        if self.notebook.tab(self.notebook.select(), "text") == '🖼️ Upload Photo':
            self.display_static_image(self.image_input_cv2.copy())

    def change_accessory_rotation(self, delta):
        """Rotates the selected accessory by 'delta' degrees (Q/E key)."""
        if not self.selected_accessory_name:
            return

        override = self.accessory_overrides[self.selected_accessory_name]

        # If offset is None, initialize it to 0 (meaning no displacement from face tilt)
        if override.get('rotation_offset') is None:
            override['rotation_offset'] = 0

        # Apply the rotation delta
        override['rotation_offset'] += delta

        print(f"Rotated {self.selected_accessory_name} by {delta}°. New offset: {override['rotation_offset']:.1f}")

        # Trigger redraw
        if self.notebook.tab(self.notebook.select(), "text") == '🖼️ Upload Photo':
            self.display_static_image(self.image_input_cv2.copy())

    def clear_accessory_override(self, name):
        """Resets the position, scale, and rotation of the accessory to automatic fitting."""
        if name and name in self.accessory_overrides:
            # Reset to the initial state (None for position, 1.0 for scale, 0.0 for rotation)
            self.accessory_overrides[name] = {'x_offset': None, 'y_offset': None, 'scale_factor': 1.0, 'rotation_offset': 0.0}
            print(f"Accessory '{name}' position, scale, and rotation reset to automatic fitting.")

            # Trigger redraw
            if self.notebook.tab(self.notebook.select(), "text") == '🖼️ Upload Photo':
                self.display_static_image(self.image_input_cv2.copy())

    # --- NEW: SAVE FAVORITE LOOK FEATURE ---
    def save_favorite_look(self):
        """Saves the current displayed frame (with accessories) and the AI suggestions."""

        # Determine the source frame based on the active tab
        active_tab = self.notebook.tab(self.notebook.select(), "text")
        frame_to_save = None
        source_type = None

        if active_tab == '📹 Live Webcam':
            # Use the last processed frame from the webcam thread
            frame_to_save = self.last_frame_processed
            source_type = "Webcam"

        elif active_tab == '🖼️ Upload Photo' and self.image_input_cv2 is not None:
            # Re-process the original image to ensure accessories are applied on the saved copy
            temp_frame = self.image_input_cv2.copy()
            temp_frame = cv2.resize(temp_frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

            face, landmarks = detect_and_get_landmarks(temp_frame)

            if landmarks:
                # Apply accessories directly to the frame for saving
                for name, img_data in self.selected_accessories.items():
                    placement = determine_placement(name)
                    overrides = self.accessory_overrides.get(name)
                    temp_frame = overlay_accessory(temp_frame, img_data, landmarks, placement, overrides)

            frame_to_save = temp_frame
            source_type = "Upload"

        else:
            messagebox.showinfo("Save Failed", "No active image to save. Start the webcam or upload a photo first.")
            return

        if frame_to_save is None:
            messagebox.showinfo("Save Failed", "No frame data available to save.")
            return

        # 1. Create the 'Favorites' directory if it doesn't exist
        save_dir = "Favorites"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 2. Generate a unique filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        face_shape = self.current_face_shape.get().replace(' ', '_').replace('/', '-')

        base_filename = f"Look_{timestamp}_{face_shape}"
        image_path = os.path.join(save_dir, f"{base_filename}.png")
        text_path = os.path.join(save_dir, f"{base_filename}_Advice.txt")

        try:
            # 3. Save the image (OpenCV frame)
            cv2.imwrite(image_path, frame_to_save)

            # 4. Save the advice/suggestions to a text file
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(f"--- FACEFIT FAVORITE LOOK ---\n")
                f.write(f"Date Saved: {time.ctime()}\n")
                f.write(f"Source: {source_type}\n")
                f.write(f"Detected Face Shape: {self.current_face_shape.get()}\n")
                f.write(f"Gender Profile: {self.gender_var.get()}\n")
                f.write("\n--- Applied Accessories ---\n")
                if self.selected_accessories:
                    for name in self.selected_accessories:
                        f.write(f"- {name}\n")
                        # Include user overrides in the text file for documentation
                        override = self.accessory_overrides.get(name, {})
                        f.write(f"  > Scale: {override.get('scale_factor', 1.0):.2f}, X-Offset: {override.get('x_offset', 'Auto')}, Y-Offset: {override.get('y_offset', 'Auto')}, Rotation Offset: {override.get('rotation_offset', 0.0):.1f} deg\n")
                else:
                    f.write("No accessories applied.\n")

                f.write("\n--- Personalized Suggestions ---\n")
                if self.ai_suggestions:
                    for title, data in self.ai_suggestions.items():
                        f.write(f"\n{title}:\n")
                        f.write(f"{data.get('text', 'No advice available.')}\n")
                else:
                    f.write("No advice currently displayed.\n")

            messagebox.showinfo("Save Successful",
                                f"Favorite Look saved to the 'Favorites' folder:\n{base_filename}.png and Advice.txt")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save files: {e}")

    # --- REMAINING I/O METHODS (Webcam/File) ---
    def start_webcam(self):
        global CAP, LIVE_STREAMING
        if not LIVE_STREAMING:
            try:
                CAP = cv2.VideoCapture(0)
                CAP.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
                CAP.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
                LIVE_STREAMING = True
                self.frame_count = 0
                self.process_webcam_frame()
                self.video_label.config(text="Live try-on active. Use Arrow Keys to move, +/- to resize, and Q/E to rotate selected accessory.")
            except Exception as e:
                messagebox.showerror("Camera Error", f"Could not open webcam: {e}")

    def stop_webcam(self):
        global CAP, LIVE_STREAMING
        if LIVE_STREAMING and CAP is not None:
            CAP.release()
            CAP = None
        LIVE_STREAMING = False
        self.last_landmarks = None
        self.tracker_active = False # Reset tracker state
        if self.notebook.tab(self.notebook.select(), "text") == '📹 Live Webcam':
             self.video_label.config(image='', text="Click 'Start Webcam' to begin live try-on. Use Arrow Keys to move, +/- to resize, and Q/E to rotate selected accessory.")

    def load_image_file(self):
        self.stop_webcam()
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.webp")]
        )
        if file_path:
            self.image_input_cv2 = cv2.imread(file_path)

            if self.image_input_cv2 is None:
                 messagebox.showerror("File Error", "Failed to load image file.")
                 return

            self.display_static_image(self.image_input_cv2.copy())

    def load_selected_accessories(self, event=None):
        self.selected_accessories = {}
        selected_indices = self.accessory_listbox.curselection()

        current_selection_names = [] # Track names for cleanup

        for i in selected_indices:
            filename = self.accessory_listbox.get(i)
            current_selection_names.append(filename) # Keep track of currently loaded

            placement = determine_placement(filename)
            try:
                with open(os.path.join(ACCESORY_DIR, filename), 'rb') as f:
                    img_data = f.read()

                processed_img = process_accessory_image(img_data, placement)
                if processed_img is not None:
                    self.selected_accessories[filename] = processed_img

                    # Initialize override if it doesn't exist
                    if filename not in self.accessory_overrides:
                        # x_offset/y_offset=None: auto-placement
                        # scale_factor=1.0: auto-scaling
                        # rotation_offset=0.0: only face tilt
                        self.accessory_overrides[filename] = {'x_offset': None, 'y_offset': None, 'scale_factor': 1.0, 'rotation_offset': 0.0}

            except Exception as e:
                print(f"Error loading accessory {filename}: {e}")

        # Clean up overrides for accessories that are no longer selected
        names_to_remove = [name for name in self.accessory_overrides if name not in current_selection_names]
        for name in names_to_remove:
            del self.accessory_overrides[name]

        # Automatically select the last accessory added for editing
        if current_selection_names:
            self.selected_accessory_name = current_selection_names[-1]
        else:
            self.selected_accessory_name = None

        if self.image_input_cv2 is not None and self.notebook.tab(self.notebook.select(), "text") == '🖼️ Upload Photo':
            self.display_static_image(self.image_input_cv2.copy())

    def display_static_image(self, frame):
        """Processes and displays the static image with face detection and accessories."""

        frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

        face, landmarks = detect_and_get_landmarks(frame)

        if landmarks:
            face_shape = classify_face_shape(landmarks)
            self.current_face_shape.set(face_shape)

            landmarks_np = np.array(landmarks)
            x = min(landmarks_np[:, 0])
            y = min(landmarks_np[:, 1])
            x_max = max(landmarks_np[:, 0])
            y_max = max(landmarks_np[:, 1])

            cv2.rectangle(frame, (x, y), (x_max, y_max), hex_to_bgr(COLORS['Dark_Olive']), 2)

            for name, img_data in self.selected_accessories.items():
                placement = determine_placement(name)
                # --- Pass the override data ---\
                overrides = self.accessory_overrides.get(name)
                frame = overlay_accessory(frame, img_data, landmarks, placement, overrides)

        else:
            self.current_face_shape.set("Face Not Found")
            messagebox.showinfo("Detection Status", "Could not detect a full face in the uploaded image.")

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.static_image_label.imgtk = imgtk
        self.static_image_label.configure(image=imgtk, text='')

        self.update_suggestions()

if __name__ == "__main__":
    app = FaceFitApp()
    app.mainloop()