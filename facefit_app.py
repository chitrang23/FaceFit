import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import dlib
import numpy as np
import pickle
import os
from PIL import Image, ImageTk
import math

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
# NOTE: These paths must exist for the app to function properly.
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
CLASSIFIER_MODEL_PATH = "face_shape_classifier.pkl"
ACCESORY_DIR = "accessories"

# --- GLOBAL VARIABLES ---
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
CAP = None
LIVE_STREAMING = False
LAST_FACE_SHAPE = "Unknown"

# --- ACCESSORY GROUPING AND PLACEMENT LOGIC ---
ACCESSORY_KEYWORDS = {
    'Glasses / Sunglasses': ['glass', 'sun', 'spectacle', 'eye', 'frame'],
    'Hats / Headwear': ['cap', 'hat', 'fedora', 'beanie', 'headband', 'crown', 'brim'],
    'Earrings / Jewelry': ['earring', 'hoop', 'stud', 'jewel', 'pendant', 'drop'],
    'Necklaces / Scarf': ['necklace', 'chain', 'pendant', 'scarf', 'tie', 'choker'],
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
    if any(keyword in name for keyword in ACCESSORY_KEYWORDS['Necklaces / Scarf']):
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

def process_accessory_image(img_data):
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

    WHITE_THRESHOLD = 240
    b, g, r, _ = cv2.split(bgra_img)
    mask_white = (b > WHITE_THRESHOLD) & (g > WHITE_THRESHOLD) & (r > WHITE_THRESHOLD)
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


def overlay_accessory(frame, accessory_img, landmarks, placement='default'):
    if PREDICTOR is None or len(np.array(landmarks)) < 68 or accessory_img.shape[2] != 4:
        return frame
        
    landmarks_np = np.array(landmarks)
    
    # --- 1. Calculate Face Roll Angle ---
    p_left_eye_outer = landmarks_np[36] 
    p_right_eye_outer = landmarks_np[45] 
    dx = p_right_eye_outer[0] - p_left_eye_outer[0]
    dy = p_right_eye_outer[1] - p_left_eye_outer[1]
    angle_deg = np.degrees(np.arctan2(dy, dx))
    
    # --- 2. Calculate Scaling and Center Points ---
    p_left_temple = landmarks_np[0]
    p_right_temple = landmarks_np[16] 
    p_nose_bridge = landmarks_np[27]
    p_chin = landmarks_np[8] 
    p_left_eye_inner = landmarks_np[39] 
    p_right_eye_inner = landmarks_np[42] 

    face_width = np.linalg.norm(p_right_temple - p_left_temple)
    center_x_face = (p_left_temple[0] + p_right_temple[0]) // 2
    
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
        target_width = int(eye_distance * 1.8) 
        target_height = int((target_width / w_acc_orig) * h_acc_orig)
    else:
        target_width = int(face_width * target_width_multiplier)
        target_height = int((target_width / w_acc_orig) * h_acc_orig)
    
    if target_width < 10 or target_height < 10: return frame 

    # --- 3. Rotate and Resize the Accessory ---
    rotated_acc, new_w, new_h = rotate_accessory(accessory_img, angle_deg, target_width, target_height)

    # --- 4. Positioning Logic ---
    
    if placement == 'glasses' or placement == 'default':
        center_x_target = (p_left_eye_inner[0] + p_right_eye_inner[0]) // 2 
        center_y_eye_line = (landmarks_np[36][1] + landmarks_np[45][1]) // 2
        center_y_target = center_y_eye_line + int(target_height * 0.10) 
        
        x_offset = center_x_target - int(new_w / 2)
        y_offset = center_y_target - int(new_h / 2) 
        
    elif placement == 'hat':
        center_y_face = p_nose_bridge[1] - int(face_width * 0.5) 
        
        x_offset = center_x_face - int(new_w / 2)
        y_offset = center_y_face - int(new_h / 2)
        
    elif placement == 'earrings':
        # Left Earring
        earring_center_x_left = p_left_temple[0]
        earring_center_y_left = p_left_temple[1] + int(face_width * 0.05) 
        
        x_offset_left = earring_center_x_left - int(new_w / 2)
        y_offset_left = earring_center_y_left - int(new_h / 2)
        
        frame = _apply_single_overlay(frame, rotated_acc, x_offset_left, y_offset_left)
        
        # Right Earring
        earring_center_x_right = p_right_temple[0]
        earring_center_y_right = p_right_temple[1] + int(face_width * 0.05)
        
        x_offset_right = earring_center_x_right - int(new_w / 2)
        y_offset_right = earring_center_y_right - int(new_h / 2)
        
        return _apply_single_overlay(frame, rotated_acc, x_offset_right, y_offset_right)

    elif placement == 'necklace':
        neck_y_start = p_chin[1] 
        
        x_offset = center_x_face - int(new_w / 2)
        y_offset = neck_y_start 
        
    return _apply_single_overlay(frame, rotated_acc, x_offset, y_offset)


def _apply_single_overlay(frame, accessory_img_resized, x_offset, y_offset):
    h_acc, w_acc, _ = accessory_img_resized.shape
    
    y1, y2 = y_offset, y_offset + h_acc
    x1, x2 = x_offset, x_offset + w_acc

    y1_frame, x1_frame = max(0, y1), max(0, x1)
    y2_frame, x2_frame = min(frame.shape[0], y2), min(frame.shape[1], x2)

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

def classify_face_shape(landmarks):
    # Classification logic (unchanged)
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

# --- Accessory Database (Unchanged) ---
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


def get_accessory_suggestion(face_shape, gender):
    suggestions = {
        'Optical Frames': ACCESSORIES_DB.get(face_shape, ACCESSORIES_DB['Unknown'])['Optical Frames'],
        'Sun & Protective Wear': ACCESSORIES_DB.get(face_shape, ACCESSORIES_DB['Unknown'])['Sun & Protective Wear'],
        'Hats & Headwear': ACCESSORIES_DB.get(face_shape, ACCESSORIES_DB['Unknown'])['Hats & Headwear'],
        'Earrings & Studs': ACCESSORIES_DB.get(face_shape, ACCESSORIES_DB['Unknown'])['Earrings & Studs'],
        'Necklaces & Pendants': ACCESSORIES_DB.get(face_shape, ACCESSORIES_DB['Unknown'])['Necklaces & Pendants'],
    }
    
    if gender == 'Male':
        suggestions['Earrings & Studs'] = {'text': 'N/A (Filtered by Male Profile, unless desired)'}
        suggestions['Hats & Headwear'] = {'text': suggestions['Hats & Headwear']['text'].replace("Floppy", "Structured")}
        
    return suggestions

# --- Tkinter Application Class ---
class FaceFitApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("FaceFit AI: Virtual Try-On")
        self.geometry("1000x650") 
        
        self.configure_styles() # <-- Style configuration
        
        self.gender_var = tk.StringVar(value='Female')
        self.category_var = tk.StringVar(value=list(ACCESSORY_CATEGORIES.keys())[0] if ACCESSORY_CATEGORIES else "")
        self.accessory_files = []
        self.current_face_shape = tk.StringVar(value="N/A")
        self.image_input_cv2 = None 
        self.accessory_listbox_var = tk.StringVar()

        # Set main window background
        self.tk_setPalette(background=COLORS['White'])

        main_paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Use Custom Frame Style
        left_frame = ttk.Frame(main_paned, width=650, style='Custom.TFrame')
        main_paned.add(left_frame, weight=3)
        self.setup_display_tabs(left_frame)
        
        right_frame = ttk.Frame(main_paned, width=300, style='Custom.TFrame')
        main_paned.add(right_frame, weight=1)
        
        self.setup_controls(right_frame)
        self.setup_results(right_frame)
        
        if DLIB_LOAD_ERROR:
            messagebox.showerror("Initialization Error", "Failed to load dlib files or models. Face detection will be limited.")

    def configure_styles(self):
        s = ttk.Style()
        s.theme_use('default') 
        
        # --- General Styles ---
        # Base frame is White
        s.configure('Custom.TFrame', background=COLORS['White'])
        
        # 🟢 FIX 1: New style for the Suggestions Frame
        s.configure('LightPink.TFrame', background=COLORS['Light_Pink'])
        
        # Base label style is White background
        s.configure('Custom.TLabel', background=COLORS['White'], foreground=COLORS['Black'], font=('Arial', 10))
        # Use the bold, deep color for header labels (White background)
        s.configure('Header.TLabel', background=COLORS['White'], foreground=COLORS['Deep_Indigo'], font=('Arial', 10, 'bold'))
        # Use Light_Pink background for the result display (the shape label itself)
        s.configure('Result.TLabel', background=COLORS['Light_Pink'], foreground=COLORS['Deep_Indigo'], font=('Arial', 14, 'bold'), padding=5)
        
        # --- Notebook (Tabs) Style ---
        s.configure('TNotebook', background=COLORS['White'], borderwidth=0)
        s.configure('TNotebook.Tab', background=COLORS['Light_Pink'], foreground=COLORS['Black'], padding=[10, 5])
        s.map('TNotebook.Tab', 
              background=[('selected', COLORS['Deep_Indigo']), ('active', COLORS['Light_Pink'])],
              foreground=[('selected', COLORS['White']), ('active', COLORS['Black'])],
              expand=[('selected', [1, 1, 1, 0])])
              
        # --- Button Styles ---
        s.configure('Primary.TButton', background=COLORS['Deep_Indigo'], foreground=COLORS['White'], font=('Arial', 10, 'bold'), borderwidth=0)
        s.map('Primary.TButton', background=[('active', COLORS['Bright_Violet']), ('!disabled', COLORS['Deep_Indigo'])])
        
        s.configure('Success.TButton', background=COLORS['Aqua_Green'], foreground=COLORS['Black'], font=('Arial', 10, 'bold'), borderwidth=0)
        s.map('Success.TButton', background=[('active', COLORS['Dark_Olive']), ('!disabled', COLORS['Aqua_Green'])])
        
        s.configure('Danger.TButton', background=COLORS['Crimson_Red'], foreground=COLORS['White'], font=('Arial', 10, 'bold'), borderwidth=0)
        s.map('Danger.TButton', background=[('active', COLORS['Light_Pink']), ('!disabled', COLORS['Crimson_Red'])])
        
        # --- LabelFrame Style ---
        s.configure('TLabelFrame', background=COLORS['White'], bordercolor=COLORS['Dark_Olive'])
        s.configure('TLabelFrame.Label', background=COLORS['White'], foreground=COLORS['Deep_Indigo'], font=('Arial', 11, 'bold'))

        # --- Combobox (Set background/foreground for general visibility) ---
        s.configure('TCombobox', fieldbackground=COLORS['White'], background=COLORS['White'], foreground=COLORS['Black'])


    def setup_display_tabs(self, parent_frame):
        self.notebook = ttk.Notebook(parent_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.webcam_tab = ttk.Frame(self.notebook, style='Custom.TFrame')
        self.notebook.add(self.webcam_tab, text='📹 Live Webcam')
        self.webcam_display_frame = ttk.Frame(self.webcam_tab, style='Custom.TFrame')
        self.webcam_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.video_label = ttk.Label(self.webcam_display_frame, text="Click 'Start Webcam' to begin live try-on.", style='Custom.TLabel', anchor='center')
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        self.upload_tab = ttk.Frame(self.notebook, style='Custom.TFrame')
        self.notebook.add(self.upload_tab, text='🖼️ Upload Photo')
        self.upload_display_frame = ttk.Frame(self.upload_tab, style='Custom.TFrame')
        self.upload_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Use Primary Button Style
        ttk.Button(self.upload_display_frame, text="1. Select Photo for Try-On", command=self.load_image_file, style='Primary.TButton').pack(fill=tk.X, pady=5)
        
        self.static_image_label = ttk.Label(self.upload_display_frame, text="Uploaded image will appear here.", style='Custom.TLabel', anchor='center')
        self.static_image_label.pack(fill=tk.BOTH, expand=True)
        
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

    def on_tab_change(self, event):
        selected_tab = self.notebook.tab(self.notebook.select(), "text")
        if selected_tab != '📹 Live Webcam':
            self.stop_webcam()
            
    def setup_controls(self, parent_frame):
        # Use TLabelFrame
        profile_group = ttk.LabelFrame(parent_frame, text="User Profile", padding="10 10 10 10")
        profile_group.pack(fill=tk.X, padx=5, pady=5)
        
        # Use Custom.TLabel
        ttk.Label(profile_group, text="Gender:", style='Custom.TLabel').grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.Combobox(profile_group, textvariable=self.gender_var, 
                     values=['Female', 'Male', 'Other', 'Prefer not to say'], state='readonly').grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        self.gender_var.trace_add("write", lambda *args: self.update_suggestions())

        accessory_group = ttk.LabelFrame(parent_frame, text="Select Accessories (Multi-Select)", padding="10 10 10 10")
        accessory_group.pack(fill=tk.X, padx=5, pady=5)
        
        # Use Custom.TLabel
        ttk.Label(accessory_group, text="Category:", style='Custom.TLabel').pack(fill=tk.X, padx=5, pady=2)
        
        self.category_dropdown = ttk.Combobox(accessory_group, 
                                              textvariable=self.category_var, 
                                              values=list(ACCESSORY_CATEGORIES.keys()), 
                                              state='readonly')
        self.category_dropdown.pack(fill=tk.X, padx=5, pady=2)
        self.category_dropdown.bind('<<ComboboxSelected>>', self.update_accessory_listbox)
        
        # Use Custom.TLabel
        ttk.Label(accessory_group, text="Files (Ctrl/Shift):", style='Custom.TLabel').pack(fill=tk.X, padx=5, pady=2)
        
        # Apply custom Listbox colors (needs Tkinter Listbox, not Ttk, for complex styling)
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
        
        # Use custom button styles
        ttk.Button(webcam_group, text="Start Webcam", command=self.start_webcam, style='Success.TButton').grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        ttk.Button(webcam_group, text="Stop Webcam", command=self.stop_webcam, style='Danger.TButton').grid(row=0, column=1, sticky='ew', padx=5, pady=5)


    # In FaceFitApp class:

# ... (Previous code) ...

    def setup_results(self, parent_frame):
        results_group = ttk.LabelFrame(parent_frame, text="Results & Suggestions", padding="10 10 10 10")
        results_group.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Use Header.TLabel
        ttk.Label(results_group, text="Detected Face Shape:", style='Header.TLabel').grid(row=0, column=0, sticky='w', padx=5, pady=5)
        # Use Result.TLabel
        self.shape_label = ttk.Label(results_group, textvariable=self.current_face_shape, style='Result.TLabel')
        self.shape_label.grid(row=0, column=1, sticky='e', padx=5, pady=5)
        
        # --- SCROLLABLE SUGGESTIONS AREA (New Implementation) ---
        
        # 1. Create a Canvas to hold the scrollable content
        self.suggestions_canvas = tk.Canvas(results_group, 
                                            bg=COLORS['Light_Pink'], # Canvas background matches frame style
                                            highlightthickness=0) # Remove default canvas border
        self.suggestions_canvas.grid(row=1, column=0, columnspan=2, sticky='nsew', pady=10)
        
        # 2. Create the Scrollbar and link it to the Canvas
        self.suggestions_scrollbar = ttk.Scrollbar(results_group, orient="vertical", command=self.suggestions_canvas.yview)
        self.suggestions_scrollbar.grid(row=1, column=2, sticky='ns', pady=10) # Place scrollbar next to canvas
        
        # 3. Configure the Canvas
        self.suggestions_canvas.configure(yscrollcommand=self.suggestions_scrollbar.set)
        
        # 4. Create the Suggestions Frame (The content container)
        # This frame is what will hold all the suggestion labels. It is placed INSIDE the canvas.
        self.suggestions_frame = ttk.Frame(self.suggestions_canvas, style='LightPink.TFrame', padding=5)
        
        # Create a window inside the canvas to hold the frame
        self.suggestions_canvas_window = self.suggestions_canvas.create_window(
            (0, 0), window=self.suggestions_frame, anchor="nw", tags="self.suggestions_frame"
        )

        # --- Configure expansion and binding for scrollbar ---
        results_group.grid_rowconfigure(1, weight=1)
        results_group.grid_columnconfigure(0, weight=1)
        
        # Bind the frame size and canvas scroll region
        self.suggestions_frame.bind("<Configure>", self._on_suggestions_frame_configure)
        self.suggestions_canvas.bind('<Configure>', self._on_suggestions_canvas_configure)
        
        self.update_suggestions()
        
    def _on_suggestions_frame_configure(self, event):
        """Update the scroll region of the canvas when the inner frame changes size."""
        self.suggestions_canvas.configure(scrollregion=self.suggestions_canvas.bbox("all"))

    def _on_suggestions_canvas_configure(self, event):
        """Update the inner frame's width to match the canvas's width."""
        canvas_width = event.width
        self.suggestions_canvas.itemconfig(self.suggestions_canvas_window, width=canvas_width)

# ... (The rest of the class methods) ...    
    def update_accessory_listbox(self, event=None):
        selected_category = self.category_var.get()
        self.accessory_listbox.delete(0, tk.END)
        
        if selected_category in ACCESSORY_CATEGORIES:
            for item in ACCESSORY_CATEGORIES[selected_category]:
                self.accessory_listbox.insert(tk.END, item)
        
        self.accessory_files = [] 
        if self.image_input_cv2 is not None and not LIVE_STREAMING:
            self.run_static_try_on(self.image_input_cv2.copy())

    def load_selected_accessories(self, event=None):
        selected_indices = self.accessory_listbox.curselection()
        selected_files = [self.accessory_listbox.get(i) for i in selected_indices]
        
        self.accessory_files = [] 
        
        for selected_file in selected_files:
            try:
                accessory_file_path = os.path.join(ACCESORY_DIR, selected_file)
                with open(accessory_file_path, "rb") as f:
                    accessory_data_raw = f.read()
                
                accessory_img_bgra = process_accessory_image(accessory_data_raw)
                
                if accessory_img_bgra is not None:
                    placement = determine_placement(selected_file)
                    self.accessory_files.append({
                        'name': selected_file,
                        'img_data': accessory_img_bgra,
                        'placement': placement
                    })
                
            except Exception as e:
                print(f"Failed to load {selected_file}: {e}")
                
        if self.image_input_cv2 is not None and not LIVE_STREAMING:
            self.run_static_try_on(self.image_input_cv2.copy())
            
    def load_image_file(self):
        self.stop_webcam()
        self.notebook.select(self.upload_tab)
        
        filepath = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg")]
        )
        if not filepath:
            return

        try:
            image_cv2 = cv2.imread(filepath)
            if image_cv2 is None:
                messagebox.showerror("Image Error", "Could not read the image file.")
                return

            self.image_input_cv2 = image_cv2
            self.run_static_try_on(image_cv2.copy())
            
        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred during image loading: {e}")

    def run_static_try_on(self, image_cv2):
        
        h, w = image_cv2.shape[:2]
        scale = min(VIDEO_WIDTH / w, VIDEO_HEIGHT / h)
        new_w, new_h = int(w * scale), int(h * scale)
        image_cv2 = cv2.resize(image_cv2, (new_w, new_h))
        
        face, landmarks = detect_and_get_landmarks(image_cv2)
        
        face_shape = "N/A"
        processed_image = image_cv2
        
        if face is not None:
            face_shape = classify_face_shape(landmarks)
            processed_image = image_cv2.copy()
            
            for acc in self.accessory_files:
                processed_image = overlay_accessory(processed_image, acc['img_data'], landmarks, acc['placement'])
            
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            # Use a dark contrasting color for the bounding box (Dark_Olive)
            cv2.rectangle(processed_image, (x, y), (x + w, y + h), tuple(int(COLORS['Dark_Olive'][i:i+2], 16) for i in (1, 3, 5))[::-1], 2)
        
        suggestions = get_accessory_suggestion(face_shape, self.gender_var.get())
        
        self.current_face_shape.set(face_shape)
        self.update_suggestions(suggestions)
        
        img_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        self.static_image_label.config(image=img_tk)
        self.static_image_label.image = img_tk
        
    def start_webcam(self):
        global CAP, LIVE_STREAMING
        if LIVE_STREAMING:
            return
            
        self.notebook.select(self.webcam_tab)
        self.image_input_cv2 = None 
        
        CAP = cv2.VideoCapture(0)
        if not CAP.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam. Check if another app is using it.")
            return

        CAP.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        CAP.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
        
        LIVE_STREAMING = True
        self.process_webcam_frame()

    def stop_webcam(self):
        global CAP, LIVE_STREAMING
        LIVE_STREAMING = False
        if CAP is not None:
            CAP.release()
            CAP = None
            self.video_label.config(text="Webcam stopped. Click 'Start Webcam' to resume.")

    def process_webcam_frame(self):
        global CAP, LIVE_STREAMING, LAST_FACE_SHAPE
        if not LIVE_STREAMING or CAP is None:
            return

        ret, frame = CAP.read()
        if not ret:
            self.stop_webcam()
            return

        processed_frame = frame.copy()
        
        face_shape = "N/A"
        suggestions = get_accessory_suggestion("Unknown", self.gender_var.get())
        
        if not DLIB_LOAD_ERROR:
            face, landmarks = detect_and_get_landmarks(processed_frame)

            if face is not None:
                face_shape = classify_face_shape(landmarks)
                LAST_FACE_SHAPE = face_shape
                suggestions = get_accessory_suggestion(face_shape, self.gender_var.get())
                
                for acc in self.accessory_files:
                    processed_frame = overlay_accessory(processed_frame, acc['img_data'], landmarks, acc['placement'])
                
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                # Use Dark_Olive for the bounding box
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), tuple(int(COLORS['Dark_Olive'][i:i+2], 16) for i in (1, 3, 5))[::-1], 2)
            else:
                LAST_FACE_SHAPE = "No Face"

        self.current_face_shape.set(face_shape)
        self.update_suggestions(suggestions)
        
        img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        self.video_label.config(image=img_tk)
        self.video_label.image = img_tk
        
        self.after(30, self.process_webcam_frame)
        
    def update_suggestions(self, suggestions=None):
        if suggestions is None:
            suggestions = get_accessory_suggestion(self.current_face_shape.get(), self.gender_var.get())

        for widget in self.suggestions_frame.winfo_children():
            widget.destroy()

        row = 0
        for title, item in suggestions.items():
            
            # --- 1. Title Label ---
            ttk.Label(self.suggestions_frame, 
                      text=f"• {title}:", # Use a bullet point for clarity
                      style='Header.TLabel', 
                      background=COLORS['Light_Pink'],
                      foreground=COLORS['Deep_Indigo']
            # Make the title label span both virtual columns
            ).grid(row=row, column=0, sticky='nw', padx=5, pady=(5, 0), columnspan=2)
            
            row += 1
            
            # --- 2. Text Label ---
            ttk.Label(self.suggestions_frame, 
                      text=item.get('text', 'N/A'), 
                      # Increase wraplength significantly to prevent clipping
                      wraplength=270, 
                      justify=tk.LEFT, 
                      style='Custom.TLabel', 
                      background=COLORS['Light_Pink'],
                      foreground=COLORS['Black']
            # Make the text label span both virtual columns (0 and 1)
            ).grid(row=row, column=0, sticky='w', padx=15, pady=(0, 5), columnspan=2) 
            
            row += 1

        # --- Frame Expansion Configuration ---
        # Make the suggestions frame's first column (column 0) expand to fill the available width.
        self.suggestions_frame.grid_columnconfigure(0, weight=1)
        
        # Ensure there's a row configured to take up any extra vertical space
# --- Main Execution ---
if __name__ == "__main__":
    app = FaceFitApp()
    app.mainloop()