import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import dlib
import numpy as np
import pickle
import os
from PIL import Image, ImageTk
import math

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

# --- ACCESSORY GROUPING AND PLACEMENT LOGIC ---

# Define keywords for automatic grouping
ACCESSORY_KEYWORDS = {
    'Glasses / Sunglasses': ['glass', 'sun', 'spectacle', 'eye'],
    'Hats / Headwear': ['cap', 'hat', 'fedora', 'beanie', 'headband', 'crown'],
    'Earrings / Jewelry': ['earring', 'hoop', 'stud', 'jewel', 'pendant'],
    'Necklaces / Scarf': ['necklace', 'chain', 'pendant', 'scarf', 'tie'],
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

# Global dictionary to store files categorized by group
ACCESSORY_CATEGORIES = {}

def organize_accessories():
    """Reads the accessory directory and organizes files into categories."""
    global ACCESSORY_CATEGORIES
    ACCESSORY_CATEGORIES = {cat: [] for cat in ACCESSORY_KEYWORDS.keys()}
    
    if os.path.exists(ACCESORY_DIR):
        accessory_files = sorted([f for f in os.listdir(ACCESORY_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        for filename in accessory_files:
            assigned = False
            # Iterate through specific categories for assignment
            for category, keywords in ACCESSORY_KEYWORDS.items():
                if category == 'Others': continue
                
                if any(keyword in filename.lower() for keyword in keywords):
                    ACCESSORY_CATEGORIES[category].append(filename)
                    assigned = True
                    break
            
            if not assigned:
                ACCESSORY_CATEGORIES['Others'].append(filename)
    
    # Remove empty categories
    ACCESSORY_CATEGORIES = {k: v for k, v in ACCESSORY_CATEGORIES.items() if v}

# Run organization once at startup
organize_accessories()


# --- UTILITY FUNCTIONS (Core Logic) ---

def process_accessory_image(img_data):
    """Processes the accessory image for use in overlay."""
    if img_data is None: return None
    accessory_img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_UNCHANGED)
    if accessory_img is None: return None
    
    # Image conversion logic to BGRA
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

    # White thresholding logic to remove white backgrounds
    WHITE_THRESHOLD = 240
    b, g, r, _ = cv2.split(bgra_img)
    mask_white = (b > WHITE_THRESHOLD) & (g > WHITE_THRESHOLD) & (r > WHITE_THRESHOLD)
    bgra_img[:, :, 3][mask_white] = 0
    return bgra_img

def landmarks_to_feature_vector(landmarks):
    # Feature vector generation
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
    # Dlib detection
    if PREDICTOR is None: return None, None
    img_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    faces = DETECTOR(img_rgb, 1)
    if not faces: return None, None
    face = faces[0]
    landmarks = PREDICTOR(img_rgb, face)
    points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 68)]
    return face, points

def rotate_accessory(accessory_img, angle_deg, target_width, target_height):
    """
    Rotates and resizes the accessory image while preserving transparency.
    """
    # 1. Resize first to simplify rotation center calculation
    resized_acc = cv2.resize(accessory_img, (target_width, target_height), interpolation=cv2.INTER_AREA)

    h_acc, w_acc, _ = resized_acc.shape
    
    # Get rotation matrix (center is the middle of the accessory)
    M = cv2.getRotationMatrix2D((w_acc / 2, h_acc / 2), angle_deg, 1.0)
    
    # Calculate new bounding box (to prevent clipping)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    new_w = int((h_acc * sin) + (w_acc * cos))
    new_h = int((h_acc * cos) + (w_acc * sin))
    
    # Adjust the rotation matrix to take into account the translation 
    M[0, 2] += (new_w / 2) - (w_acc / 2)
    M[1, 2] += (new_h / 2) - (h_acc / 2)
    
    # Perform the rotation
    rotated_acc = cv2.warpAffine(resized_acc, M, (new_w, new_h), 
                                 borderMode=cv2.BORDER_CONSTANT, 
                                 borderValue=(0, 0, 0, 0)) # BORDER_CONSTANT with (0,0,0,0) ensures transparency
    
    return rotated_acc, new_w, new_h


def overlay_accessory(frame, accessory_img, landmarks, placement='default'):
    """
    Overlays accessory onto the frame with rotation based on face roll angle.
    """
    if PREDICTOR is None or len(np.array(landmarks)) < 68 or accessory_img.shape[2] != 4:
        return frame
        
    landmarks_np = np.array(landmarks)
    
    # --- 1. Calculate Face Roll Angle (Based on outer eye corners) ---
    # Points 36 (left eye outer) and 45 (right eye outer) are used for roll calculation
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

    face_width = np.linalg.norm(p_right_temple - p_left_temple)
    
    # Determine base target width
    target_width_multiplier = 1.3 # Default for glasses/hat
    if placement == 'earrings':
        target_width_multiplier = 0.4 
    if placement == 'necklace':
        target_width_multiplier = 1.5 

    h_acc_orig, w_acc_orig, _ = accessory_img.shape
    target_width = int(face_width * target_width_multiplier)
    target_height = int((target_width / w_acc_orig) * h_acc_orig)
    
    if target_width < 10 or target_height < 10: return frame 

    # --- 3. Rotate and Resize the Accessory ---
    rotated_acc, new_w, new_h = rotate_accessory(accessory_img, angle_deg, target_width, target_height)

    # --- 4. Positioning Logic (Use the center of the accessory's bounding box) ---
    
    center_x_face = (p_left_temple[0] + p_right_temple[0]) // 2
    
    if placement == 'hat':
        # Position hat centered well above the nose bridge
        center_y_face = p_nose_bridge[1] - int(face_width * 0.75) 
        
        # Offsets are calculated relative to the top-left corner of the new bounding box
        x_offset = center_x_face - int(new_w / 2)
        y_offset = center_y_face - int(new_h / 2)
        
    elif placement == 'glasses' or placement == 'default':
        # Position glasses centered on the nose bridge area
        center_y_face = p_nose_bridge[1] - int(target_height * 0.4) 
        
        x_offset = center_x_face - int(new_w / 2)
        y_offset = center_y_face - int(new_h / 2)
        
        frame = _apply_single_overlay(frame, rotated_acc, x_offset, y_offset)
        return frame # Glasses/Default use single overlay and return

    elif placement == 'earrings':
        # Earrings are handled by placing the center of the bounding box near the temple landmarks.
        
        # 1. Left Earring (Near landmark 0)
        earring_center_x_left = p_left_temple[0]
        earring_center_y_left = p_left_temple[1] + int(target_height * 0.1) 
        
        x_offset_left = earring_center_x_left - int(new_w / 2)
        y_offset_left = earring_center_y_left - int(new_h / 2)
        
        frame = _apply_single_overlay(frame, rotated_acc, x_offset_left, y_offset_left)
        
        # 2. Right Earring (Near landmark 16)
        earring_center_x_right = p_right_temple[0]
        earring_center_y_right = p_right_temple[1] + int(target_height * 0.1) 
        
        x_offset_right = earring_center_x_right - int(new_w / 2)
        y_offset_right = earring_center_y_right - int(new_h / 2)
        
        return _apply_single_overlay(frame, rotated_acc, x_offset_right, y_offset_right)

    elif placement == 'necklace':
        # Position necklace centered below the chin (landmark 8)
        neck_y_start = p_chin[1] 
        
        x_offset = center_x_face - int(new_w / 2)
        # Position Y just below the chin.
        y_offset = neck_y_start 
        
    # For all single-placement items (hat, necklace)
    return _apply_single_overlay(frame, rotated_acc, x_offset, y_offset)


def _apply_single_overlay(frame, accessory_img_resized, x_offset, y_offset):
    """Handles the actual alpha blending and clipping for a single accessory placement."""
    h_acc, w_acc, _ = accessory_img_resized.shape
    
    y1, y2 = y_offset, y_offset + h_acc
    x1, x2 = x_offset, x_offset + w_acc

    # Safety bounds check against the video frame size
    y1_frame, x1_frame = max(0, y1), max(0, x1)
    y2_frame, x2_frame = min(frame.shape[0], y2), min(frame.shape[1], x2)

    # Adjust accessory slice coordinates to match the clipped frame region
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
    # Face shape classification
    global ML_MODEL_LOADED, FACE_SHAPE_MODEL
    if not landmarks: return "Unknown"
    
    if ML_MODEL_LOADED and FACE_SHAPE_MODEL is not None:
        try:
            features = landmarks_to_feature_vector(landmarks)
            if features.size > 0:
                return FACE_SHAPE_MODEL.predict([features])[0]
        except Exception:
            pass 

    # Geometric fallback logic
    landmarks_np = np.array(landmarks)
    try:
        jaw_width = np.linalg.norm(landmarks_np[3] - landmarks_np[13])
        face_length = np.linalg.norm(landmarks_np[8] - landmarks_np[27])
    except IndexError:
        return "Unknown"

    ratio = face_length / jaw_width

    if ratio > 1.3: return "Long"
    elif ratio > 1.15: return "Oval"
    elif ratio > 1.05: return "Square"
    else: return "Round"

# --- Initialization and Error Handling ---
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

# --- Accessory Database (Suggestions) ---
ACCESSORIES_DB = {
    'Round': {
        'glasses': {'text': 'Angular/Geometric frames'},
        'sunglasses': {'text': 'Square or rectangular lenses'},
        'hats': {'text': 'Structured, angular crowns'},
        'earrings': {'text': 'Long, dangling, or geometric drops'},
        'necklaces': {'text': 'Long chains to create length'},
    },
    'Oval': {
        'glasses': {'text': 'Any frame style works well (avoid very narrow)'},
        'sunglasses': {'text': 'Oversized frames or aviators'},
        'hats': {'text': 'Wide brims or fedoras'},
        'earrings': {'text': 'Studs or teardrops (medium length)'},
        'necklaces': {'text': 'Chokers or short chains (to showcase symmetry)'},
    },
    'Square': {
        'glasses': {'text': 'Round or Oval frames (to soften angles)'},
        'sunglasses': {'text': 'Cat-eye or round frames'},
        'hats': {'text': 'Round crowns, floppy brims, soft materials'},
        'earrings': {'text': 'Round or hoop earrings'},
        'necklaces': {'text': 'Long pendants or curved shapes'},
    },
    'Heart': {
        'glasses': {'text': 'Bottom-heavy, rimless, or cat-eye (to balance chin)'},
        'sunglasses': {'text': 'Aviator or rimless styles'},
        'hats': {'text': 'Medium brim, bucket hats'},
        'earrings': {'text': 'Teardrops or chandeliers (narrow at top, wide at bottom)'},
        'necklaces': {'text': 'Short chains or chokers'},
    },
    'Long': {
        'glasses': {'text': 'Deep frames with decorative temples (add width)'},
        'sunglasses': {'text': 'Tall, round, or geometric frames'},
        'hats': {'text': 'Low crowns, wide brims'},
        'earrings': {'text': 'Round, volume-adding, or statement earrings'},
        'necklaces': {'text': 'Short, chunky necklaces or chokers (add horizontal line)'},
    },
    'Unknown': {
        'glasses': {'text': 'Try a neutral, medium size frame.'},
        'sunglasses': {'text': 'Try standard aviators.'},
        'hats': {'text': 'No specific suggestion.'},
        'earrings': {'text': 'Simple studs.'},
        'necklaces': {'text': 'Simple chain.'},
    }
}

def get_accessory_suggestion(face_shape, gender):
    suggestions = ACCESSORIES_DB.get(face_shape, ACCESSORIES_DB['Unknown']).copy()
    
    if gender == 'Male':
        # Simple filter for male profile
        suggestions['earrings'] = {'text': 'N/A (Filtered by profile)'}
        
    return suggestions

# --- Tkinter Application Class ---
class FaceFitApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("FaceFit AI: Virtual Try-On")
        self.geometry("1000x650") 
        
        self.gender_var = tk.StringVar(value='Female')
        self.category_var = tk.StringVar(value=list(ACCESSORY_CATEGORIES.keys())[0] if ACCESSORY_CATEGORIES else "")
        self.accessory_files = []
        self.current_face_shape = tk.StringVar(value="N/A")
        self.image_input_cv2 = None 
        self.accessory_listbox_var = tk.StringVar()

        main_paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = ttk.Frame(main_paned, width=650)
        main_paned.add(left_frame, weight=3)
        self.setup_display_tabs(left_frame)
        
        right_frame = ttk.Frame(main_paned, width=300)
        main_paned.add(right_frame, weight=1)
        
        self.setup_controls(right_frame)
        self.setup_results(right_frame)
        
        if DLIB_LOAD_ERROR:
            messagebox.showerror("Initialization Error", "Failed to load dlib files or models. Face detection will be limited.")

    def setup_display_tabs(self, parent_frame):
        self.notebook = ttk.Notebook(parent_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.webcam_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.webcam_tab, text='📹 Live Webcam')
        self.webcam_display_frame = ttk.Frame(self.webcam_tab)
        self.webcam_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.video_label = ttk.Label(self.webcam_display_frame, text="Click 'Start Webcam' to begin live try-on.")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        self.upload_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.upload_tab, text='🖼️ Upload Photo')
        self.upload_display_frame = ttk.Frame(self.upload_tab)
        self.upload_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        ttk.Button(self.upload_display_frame, text="1. Select Photo for Try-On", command=self.load_image_file).pack(fill=tk.X, pady=5)
        self.static_image_label = ttk.Label(self.upload_display_frame, text="Uploaded image will appear here.")
        self.static_image_label.pack(fill=tk.BOTH, expand=True)
        
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

    def on_tab_change(self, event):
        selected_tab = self.notebook.tab(self.notebook.select(), "text")
        if selected_tab != '📹 Live Webcam':
            self.stop_webcam()
            
    def setup_controls(self, parent_frame):
        profile_group = ttk.LabelFrame(parent_frame, text="User Profile", padding="10 10 10 10")
        profile_group.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(profile_group, text="Gender:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.Combobox(profile_group, textvariable=self.gender_var, 
                     values=['Female', 'Male', 'Other', 'Prefer not to say'], state='readonly').grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        self.gender_var.trace_add("write", lambda *args: self.update_suggestions())

        accessory_group = ttk.LabelFrame(parent_frame, text="Select Accessories (Multi-Select)", padding="10 10 10 10")
        accessory_group.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(accessory_group, text="Category:").pack(fill=tk.X, padx=5, pady=2)
        
        self.category_dropdown = ttk.Combobox(accessory_group, 
                                              textvariable=self.category_var, 
                                              values=list(ACCESSORY_CATEGORIES.keys()), 
                                              state='readonly')
        self.category_dropdown.pack(fill=tk.X, padx=5, pady=2)
        self.category_dropdown.bind('<<ComboboxSelected>>', self.update_accessory_listbox)
        
        ttk.Label(accessory_group, text="Files (Ctrl/Shift):").pack(fill=tk.X, padx=5, pady=2)
        
        self.accessory_listbox = tk.Listbox(accessory_group, height=6, selectmode=tk.MULTIPLE, listvariable=self.accessory_listbox_var)
        self.accessory_listbox.pack(fill=tk.X, padx=5, pady=2)
        
        self.accessory_listbox.bind('<<ListboxSelect>>', self.load_selected_accessories)
        self.update_accessory_listbox()

        webcam_group = ttk.LabelFrame(parent_frame, text="Camera Control", padding="10 10 10 10")
        webcam_group.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(webcam_group, text="Start Webcam", command=self.start_webcam).grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        ttk.Button(webcam_group, text="Stop Webcam", command=self.stop_webcam).grid(row=0, column=1, sticky='ew', padx=5, pady=5)


    def setup_results(self, parent_frame):
        results_group = ttk.LabelFrame(parent_frame, text="Results & Suggestions", padding="10 10 10 10")
        results_group.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Label(results_group, text="Detected Face Shape:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.shape_label = ttk.Label(results_group, textvariable=self.current_face_shape, font=('Arial', 12, 'bold'))
        self.shape_label.grid(row=0, column=1, sticky='e', padx=5, pady=5)
        
        self.suggestions_frame = ttk.Frame(results_group)
        self.suggestions_frame.grid(row=1, column=0, columnspan=2, sticky='nsew', pady=10)
        results_group.grid_rowconfigure(1, weight=1)
        results_group.grid_columnconfigure(0, weight=1)
        results_group.grid_columnconfigure(1, weight=1)
        self.update_suggestions()
        
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
        
        if face is None:
            face_shape = "N/A"
            processed_image = image_cv2
        else:
            face_shape = classify_face_shape(landmarks)
            processed_image = image_cv2.copy()
            
            for acc in self.accessory_files:
                processed_image = overlay_accessory(processed_image, acc['img_data'], landmarks, acc['placement'])
            
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(processed_image, (x, y), (x + w, y + h), (0, 165, 255), 2)
        
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
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
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
            ttk.Label(self.suggestions_frame, text=f"**{title.title()}:**", font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky='w', padx=5, pady=2)
            ttk.Label(self.suggestions_frame, text=item.get('text', 'N/A'), wraplength=180).grid(row=row, column=1, sticky='w', padx=5, pady=2)
            row += 1

# --- Main Execution ---
if __name__ == "__main__":
    app = FaceFitApp()
    app.mainloop()