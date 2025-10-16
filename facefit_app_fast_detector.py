import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import dlib
import numpy as np
import pickle
import os
from PIL import Image, ImageTk
import math
import threading
import time
from collections import Counter, deque

# --- IMPORTANT: PASTE YOUR GEMINI API KEY HERE ---
YOUR_GEMINI_API_KEY = "AIzaSyDwE3RsN7ZLeLKViHu_YmwNW_SJnYOfEIw"
# ----------------------------------------------------

# --- Google GenAI SDK ---
try:
    import google.genai as genai
    GEMINI_CLIENT_AVAILABLE = True
except ImportError:
    print("WARNING: google-genai not installed. AI suggestions will use fallback data.")
    GEMINI_CLIENT_AVAILABLE = False

# --- 1. CONFIGURATION & GLOBALS ---
COLORS = {'Black': '#000000', 'White': '#ffffff', 'Deep_Indigo': '#31186a', 'Crimson_Red': '#d9023f', 'Light_Pink': '#fa8caa', 'Bright_Violet': '#e78deb', 'Aqua_Green': '#6af7c1', 'Dark_Olive': '#556b2f'}
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
CLASSIFIER_MODEL_PATH = "face_shape_classifier.pkl"
ACCESORY_DIR = "accessories"
HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480
CAP, LIVE_STREAMING = None, False

# --- FINAL, CORRECT FEATURE EXTRACTION LOGIC (13 FEATURES) ---
def euclidean_distance(p1, p2): return np.linalg.norm(np.array(p1) - np.array(p2))
def calculate_angle(p1, p_center, p3):
    v1 = np.array(p1) - np.array(p_center); v2 = np.array(p3) - np.array(p_center)
    dot_product = np.dot(v1, v2); norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0: return 0.0
    return np.degrees(np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0)))

def landmarks_to_feature_vector(landmarks):
    if landmarks is None or len(landmarks) != 68: return None
    norm_factor = euclidean_distance(landmarks[36], landmarks[45])
    if norm_factor == 0: return None
    features = [
        euclidean_distance(landmarks[1], landmarks[15]) / norm_factor, euclidean_distance(landmarks[3], landmarks[13]) / norm_factor,
        euclidean_distance(landmarks[17], landmarks[26]) / norm_factor, euclidean_distance(landmarks[8], landmarks[27]) / norm_factor,
        euclidean_distance(landmarks[57], landmarks[8]) / norm_factor,
    ]
    features.extend([features[3] / features[0] if features[0] > 0 else 0, features[1] / features[0] if features[0] > 0 else 0, features[2] / features[1] if features[1] > 0 else 0])
    features.extend([calculate_angle(landmarks[2], landmarks[4], landmarks[6]), calculate_angle(landmarks[14], landmarks[12], landmarks[10]), calculate_angle(landmarks[6], landmarks[8], landmarks[10])])
    features.extend([(euclidean_distance(landmarks[50], landmarks[61]) + euclidean_distance(landmarks[52], landmarks[63])) / norm_factor, (euclidean_distance(landmarks[2], landmarks[31]) + euclidean_distance(landmarks[14], landmarks[35])) / norm_factor])
    return np.array(features)

# --- 2. ACCESSORY & PLACEMENT LOGIC ---
ACCESSORY_KEYWORDS = {
    'Glasses / Sunglasses': ['glass', 'sun', 'spectacle', 'eye', 'frame'], 'Hats / Headwear': ['cap', 'hat', 'fedora', 'beanie', 'headband', 'crown', 'brim'],
    'Earrings / Jewelry': ['earring', 'hoop', 'stud', 'jewel', 'pendant', 'drop'], 'Necklaces / Pendants': ['necklace', 'chain', 'pendant', 'scarf', 'tie', 'choker'], 'Others': []
}
def determine_placement(name):
    name = name.lower()
    for category, keywords in ACCESSORY_KEYWORDS.items():
        if category != 'Others' and any(keyword in name for keyword in keywords): return category.split(' ')[0].lower().replace('/', '')
    return 'default'

ACCESSORY_CATEGORIES = {}
def organize_accessories():
    global ACCESSORY_CATEGORIES; ACCESSORY_CATEGORIES = {cat: [] for cat in ACCESSORY_KEYWORDS.keys()}
    if os.path.exists(ACCESORY_DIR):
        for filename in sorted([f for f in os.listdir(ACCESORY_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]):
            assigned = False
            for category, keywords in ACCESSORY_KEYWORDS.items():
                if category != 'Others' and any(keyword in filename.lower() for keyword in keywords): ACCESSORY_CATEGORIES[category].append(filename); assigned = True; break
            if not assigned: ACCESSORY_CATEGORIES['Others'].append(filename)
    ACCESSORY_CATEGORIES = {k: v for k, v in ACCESSORY_CATEGORIES.items() if v}
organize_accessories()

# --- 3. CORE COMPUTER VISION & UTILITY FUNCTIONS ---
def hex_to_bgr(hex_color): return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))[::-1]
def _clamp_position(x, y, fw, fh, aw, ah): return max(0, min(x, fw - aw)), max(0, min(y, fh - ah))

def process_accessory_image(img_data):
    if img_data is None: return None
    acc_img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_UNCHANGED)
    if acc_img is None or acc_img.shape[2] != 4: return None
    b, g, r, a = cv2.split(acc_img); a[(b > 240) & (g > 240) & (r > 240)] = 0
    return cv2.merge([b, g, r, a])

def rotate_accessory(acc_img, angle, w, h):
    resized = cv2.resize(acc_img, (w, h), interpolation=cv2.INTER_AREA)
    h_acc, w_acc, _ = resized.shape; center = (w_acc // 2, h_acc // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    new_w, new_h = int((h_acc * sin) + (w_acc * cos)), int((h_acc * cos) + (w_acc * sin))
    M[0, 2] += (new_w / 2) - center[0]; M[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(resized, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)), new_w, new_h

def _apply_single_overlay(frame, acc_img, x, y):
    h_acc, w_acc, _ = acc_img.shape; frame_h, frame_w, _ = frame.shape
    x, y = _clamp_position(x, y, frame_w, frame_h, w_acc, h_acc)
    y1, y2 = y, y + h_acc; x1, x2 = x, x + w_acc
    alpha = (acc_img[:, :, 3] / 255.0)[:, :, np.newaxis]
    frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1 - alpha) + acc_img[:, :, :3] * alpha
    return frame

def overlay_accessory(frame, acc_img, landmarks, placement='default', overrides=None):
    if not landmarks or len(landmarks) < 68 or acc_img is None: return frame
    ln = np.array(landmarks); overrides = overrides or {}
    angle_deg = np.degrees(np.arctan2(ln[45][1] - ln[36][1], ln[45][0] - ln[36][0]))
    final_angle = angle_deg + overrides.get('rotation_offset', 0)
    face_width = np.linalg.norm(ln[16] - ln[0]); scale_factor = overrides.get('scale_factor', 1.0)
    h_acc, w_acc, _ = acc_img.shape
    if placement == 'glasses': target_w = int(np.linalg.norm(ln[45] - ln[36]) * 2.2 * scale_factor)
    elif placement == 'hat': target_w = int(face_width * 1.6 * scale_factor)
    elif placement == 'earrings': target_w = int(face_width * 0.35 * scale_factor)
    elif placement == 'necklace': target_w = int(np.linalg.norm(ln[11] - ln[5]) * 1.3 * scale_factor)
    else: target_w = int(face_width * 1.2 * scale_factor)
    target_h = int((target_w / w_acc) * h_acc)
    if target_w < 5 or target_h < 5: return frame
    rotated_acc, new_w, new_h = rotate_accessory(acc_img, final_angle, target_w, target_h)
    user_dx, user_dy = overrides.get('x_offset', 0), overrides.get('y_offset', 0)
    
    if placement == 'glasses': center_x, center_y = np.mean(ln[36:48], axis=0); x_pos, y_pos = int(center_x - new_w // 2), int(center_y - new_h // 2)
    elif placement == 'hat': eyebrow_top = np.mean(ln[17:27], axis=0)[1]; y_pos, x_pos = int(eyebrow_top - new_h * 0.8), int(ln[27][0] - new_w // 2)
    elif placement == 'earrings':
        frame = _apply_single_overlay(frame, rotated_acc, int(ln[2][0] - new_w*0.75 + user_dx), int(ln[2][1] + user_dy))
        return _apply_single_overlay(frame, rotated_acc, int(ln[14][0] - new_w*0.25 + user_dx), int(ln[14][1] + user_dy))
    elif placement == 'necklace': x_pos, y_pos = int(ln[8][0] - new_w // 2), int(ln[57][1])
    else: x_pos, y_pos = int(ln[30][0] - new_w // 2), int(ln[30][1] - new_h // 2)
    return _apply_single_overlay(frame, rotated_acc, x_pos + user_dx, y_pos + user_dy)

def classify_face_shape(landmarks):
    if not landmarks or len(landmarks) < 68: return "Unknown"
    if ML_MODEL_LOADED and FACE_SHAPE_MODEL is not None:
        try:
            feature_vector = landmarks_to_feature_vector(landmarks)
            if feature_vector is not None and len(feature_vector) > 0: return FACE_SHAPE_MODEL.predict([feature_vector])[0]
        except Exception as e: print(f"ML model prediction failed: {e}. Falling back.")
    return "Unknown"

# --- 4. MODEL LOADING & HYBRID DETECTION ---
DLIB_DETECTOR, FACE_CASCADE = dlib.get_frontal_face_detector(), cv2.CascadeClassifier(HAAR_CASCADE_PATH)
FACE_SHAPE_MODEL, ML_MODEL_LOADED, PREDICTOR = None, False, None
try:
    if os.path.exists(SHAPE_PREDICTOR_PATH): PREDICTOR = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    if os.path.exists(CLASSIFIER_MODEL_PATH):
        with open(CLASSIFIER_MODEL_PATH, 'rb') as f: FACE_SHAPE_MODEL = pickle.load(f); ML_MODEL_LOADED = True
except Exception as e: print(f"FATAL ERROR during Model Load: {e}")

def detect_and_get_landmarks(image_cv2, use_dlib_detector=False):
    if PREDICTOR is None: return None
    img_gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
    detector = DLIB_DETECTOR if use_dlib_detector else FACE_CASCADE
    if use_dlib_detector: faces = detector(img_gray, 1)
    else: faces = detector.detectMultiScale(img_gray, 1.1, 5, minSize=(80, 80))
    if len(faces) == 0: return None
    if use_dlib_detector: face_rect = max(faces, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))
    else: (x, y, w, h) = sorted(faces, reverse=True, key=lambda a: a[2] * a[3])[0]; face_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    shape = PREDICTOR(image_cv2, face_rect)
    return [(shape.part(i).x, shape.part(i).y) for i in range(68)]

# --- 5. AI SUGGESTIONS (BACKWARD-COMPATIBLE) ---
def generate_ai_suggestion(face_shape, gender, accessory_type):
    if not GEMINI_CLIENT_AVAILABLE or YOUR_GEMINI_API_KEY == "PASTE_YOUR_API_KEY_HERE": return f"Try classic styles for {face_shape} shape."
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = (f"As a fashion expert, give a concise, 2-sentence recommendation for '{accessory_type}' for a '{gender}' person with a '{face_shape}' face. Be specific with styles and materials.")
        return model.generate_content(prompt).text
    except AttributeError:
        print("Using legacy genai.generate_text() method.")
        return genai.generate_text(prompt=prompt).result
    except Exception as e: print(f"Gemini API Error: {e}."); return f"Try modern styles for {face_shape} shape."

# --- 6. TKINTER APPLICATION CLASS ---
class FaceFitApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs); self.title("FaceFit AI (Final Version)"); self.geometry("1000x650")
        if GEMINI_CLIENT_AVAILABLE and YOUR_GEMINI_API_KEY != "PASTE_YOUR_API_KEY_HERE":
            try: genai.configure(api_key=YOUR_GEMINI_API_KEY); print("Gemini API Client configured successfully.")
            except AttributeError: print("Old google-generativeai version detected. Key will be used implicitly.")
            except Exception as e: print(f"ERROR: Failed to configure Gemini API: {e}. Falling back.")
        else: print("WARNING: Gemini API key not provided or library not found. Falling back.")
        self.configure_styles()
        self.gender_var = tk.StringVar(value='Female'); self.category_var = tk.StringVar(value=list(ACCESSORY_CATEGORIES.keys())[0] if ACCESSORY_CATEGORIES else "")
        self.current_face_shape = tk.StringVar(value="N/A"); self.image_input_cv2, self.suggestion_thread, self.last_frame_processed = None, None, None
        self.selected_accessories = {}; self.selected_accessory_name, self.accessory_overrides = None, {}
        self.is_scanning = False; self.scan_frame_limit = 20; self.face_shape_readings = []
        # --- NEW: Landmark Smoothing ---
        self.landmark_history = deque(maxlen=5) # Store last 5 frames of landmarks
        
        main_paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL); main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left_frame = ttk.Frame(main_paned, width=650); main_paned.add(left_frame, weight=3); self.setup_display_tabs(left_frame)
        right_frame = ttk.Frame(main_paned, width=300); main_paned.add(right_frame, weight=1)
        self.setup_controls(right_frame); self.setup_results(right_frame)
        if PREDICTOR is None or FACE_CASCADE is None: messagebox.showerror("Init Error", "Critical model files are missing.")
        self.update_suggestions(); self.bind_keyboard_controls()

    def configure_styles(self):
        s = ttk.Style(); s.theme_use('clam')
        common_btn = {'font': ('Arial', 10, 'bold')}
        s.configure('TFrame', background='white'); s.configure('TLabel', background='white', font=('Arial', 10))
        s.configure('Header.TLabel', background='white', foreground=COLORS['Deep_Indigo'], font=('Arial', 10, 'bold'))
        s.configure('Result.TLabel', background=COLORS['Light_Pink'], foreground=COLORS['Deep_Indigo'], font=('Arial', 14, 'bold'), padding=5)
        s.configure('TNotebook.Tab', padding=[10, 5]); s.map('TNotebook.Tab', background=[('selected', COLORS['Deep_Indigo'])], foreground=[('selected', 'white')])
        s.configure('Primary.TButton', background=COLORS['Deep_Indigo'], foreground='white', **common_btn)
        s.configure('Success.TButton', background=COLORS['Aqua_Green'], foreground='black', **common_btn)
        s.configure('Danger.TButton', background=COLORS['Crimson_Red'], foreground='white', **common_btn)
        s.map('*.TButton', background=[('active', COLORS['Bright_Violet'])])

    def bind_keyboard_controls(self):
        for key in ['<Up>', '<Down>', '<Left>', '<Right>', '+', '-', 'q', 'e', '<Delete>', '<BackSpace>']: self.bind(key, self.handle_key_press)
    def handle_key_press(self, event):
        if event.keysym == 'Up': self.move_accessory('y', -5)
        elif event.keysym == 'Down': self.move_accessory('y', 5)
        elif event.keysym == 'Left': self.move_accessory('x', -5)
        elif event.keysym == 'Right': self.move_accessory('x', 5)
        elif event.keysym in ['plus', 'equal']: self.change_accessory_scale(0.05)
        elif event.keysym == 'minus': self.change_accessory_scale(-0.05)
        elif event.keysym == 'q': self.change_accessory_rotation(-5)
        elif event.keysym == 'e': self.change_accessory_rotation(5)
        elif event.keysym in ['Delete', 'BackSpace']: self.clear_accessory_override()
            
    def setup_display_tabs(self, parent):
        self.notebook = ttk.Notebook(parent); self.notebook.pack(fill=tk.BOTH, expand=True)
        webcam_tab = ttk.Frame(self.notebook); self.notebook.add(webcam_tab, text='📹 Live Webcam'); self.video_label = ttk.Label(webcam_tab, anchor='center'); self.video_label.pack(fill=tk.BOTH, expand=True)
        upload_tab = ttk.Frame(self.notebook); self.notebook.add(upload_tab, text='🖼️ Upload Photo'); ttk.Button(upload_tab, text="1. Select Photo", command=self.load_image_file, style='Primary.TButton').pack(fill=tk.X, pady=5); self.static_image_label = ttk.Label(upload_tab, anchor='center'); self.static_image_label.pack(fill=tk.BOTH, expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

    def on_tab_change(self, event):
        if self.notebook.tab(self.notebook.select(), "text") != '📹 Live Webcam': self.stop_webcam()

    def setup_controls(self, parent):
        profile_group = ttk.LabelFrame(parent, text="User Profile", padding=10); profile_group.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(profile_group, text="Gender:").grid(row=0, column=0, sticky='w'); gender_cb = ttk.Combobox(profile_group, textvariable=self.gender_var, values=['Female', 'Male', 'Other'], state='readonly'); gender_cb.grid(row=0, column=1, sticky='ew', padx=5); gender_cb.bind('<<ComboboxSelected>>', lambda e: self.update_suggestions())
        accessory_group = ttk.LabelFrame(parent, text="Select Accessories", padding=10); accessory_group.pack(fill=tk.X, padx=5, pady=5)
        category_cb = ttk.Combobox(accessory_group, textvariable=self.category_var, values=list(ACCESSORY_CATEGORIES.keys()), state='readonly'); category_cb.pack(fill=tk.X, pady=2); category_cb.bind('<<ComboboxSelected>>', self.update_accessory_listbox)
        self.accessory_listbox = tk.Listbox(accessory_group, height=6, selectmode=tk.MULTIPLE, exportselection=False); self.accessory_listbox.pack(fill=tk.X, pady=2); self.accessory_listbox.bind('<<ListboxSelect>>', self.load_selected_accessories); self.update_accessory_listbox()
        controls_group = ttk.LabelFrame(parent, text="Controls", padding=10); controls_group.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(controls_group, text="Start Webcam", command=self.start_webcam, style='Success.TButton').grid(row=0, column=0, sticky='ew', padx=2)
        ttk.Button(controls_group, text="Stop Webcam", command=self.stop_webcam, style='Danger.TButton').grid(row=0, column=1, sticky='ew', padx=2)
        ttk.Button(controls_group, text="Re-Scan Face Shape", command=self.start_face_scan, style='Primary.TButton').grid(row=1, column=0, sticky='ew', padx=2)
        ttk.Button(controls_group, text="💾 Save Look", command=self.save_favorite_look, style='Primary.TButton').grid(row=1, column=1, sticky='ew', padx=2)

    def update_accessory_listbox(self, event=None):
        self.accessory_listbox.delete(0, tk.END)
        for f in ACCESSORY_CATEGORIES.get(self.category_var.get(), []): self.accessory_listbox.insert(tk.END, f)
        
    def process_webcam_frame(self):
        if not LIVE_STREAMING or CAP is None: return
        ret, frame = CAP.read()
        if not ret: self.after(10, self.process_webcam_frame); return
        processed_frame = cv2.flip(cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT)), 1); self.last_frame_processed = processed_frame.copy()
        
        raw_landmarks = detect_and_get_landmarks(processed_frame, use_dlib_detector=False)
        
        if raw_landmarks:
            self.landmark_history.append(raw_landmarks)
        
        if self.landmark_history:
            # Average the landmarks
            smoothed_landmarks = np.mean(self.landmark_history, axis=0).astype(int).tolist()

            if self.is_scanning:
                self.face_shape_readings.append(classify_face_shape(smoothed_landmarks))
                if len(self.face_shape_readings) >= self.scan_frame_limit:
                    self.is_scanning = False
                    if self.face_shape_readings:
                        most_common = Counter(self.face_shape_readings).most_common(1)[0][0]
                        if most_common != "Unknown":
                            self.current_face_shape.set(most_common); self.update_suggestions()
            
            for name, img_data in self.selected_accessories.items():
                processed_frame = overlay_accessory(processed_frame, img_data, smoothed_landmarks, determine_placement(name), self.accessory_overrides.get(name))
                
        img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB); imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.video_label.imgtk = imgtk; self.video_label.configure(image=imgtk)
        self.after(10, self.process_webcam_frame)
    
    def start_face_scan(self):
        if not LIVE_STREAMING: messagebox.showinfo("Info", "Please start the webcam first."); return
        self.is_scanning = True; self.face_shape_readings = []; self.current_face_shape.set("Scanning...")

    def get_accessory_suggestion_async(self):
        face_shape, gender = self.current_face_shape.get(), self.gender_var.get()
        suggestions = {atype: {'text': generate_ai_suggestion(face_shape, gender, atype)} for atype in ACCESSORY_KEYWORDS if atype != 'Others'}
        self.after(0, lambda: self.display_suggestions(suggestions))

    def update_suggestions(self):
        for widget in self.suggestions_frame.winfo_children(): widget.destroy()
        if self.current_face_shape.get() not in ["N/A", "Face Not Found", "Scanning..."]:
            ttk.Label(self.suggestions_frame, text="🧠 Calling Gemini for suggestions...", wraplength=270).pack(fill=tk.X, padx=5)
            if self.suggestion_thread is None or not self.suggestion_thread.is_alive():
                self.suggestion_thread = threading.Thread(target=self.get_accessory_suggestion_async); self.suggestion_thread.daemon = True; self.suggestion_thread.start()
        else: ttk.Label(self.suggestions_frame, text="Start webcam or upload photo.", wraplength=270).pack(fill=tk.X, padx=5)

    def setup_results(self, parent):
        results_group = ttk.LabelFrame(parent, text="Results & Suggestions", padding=10); results_group.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        ttk.Label(results_group, text="Detected Face Shape:", style='Header.TLabel').pack(fill=tk.X)
        ttk.Label(results_group, textvariable=self.current_face_shape, style='Result.TLabel').pack(fill=tk.X, pady=5)
        ttk.Label(results_group, text="Gemini AI Fit Advice:", style='Header.TLabel').pack(fill=tk.X, pady=5)
        self.suggestions_frame = ttk.Frame(results_group); self.suggestions_frame.pack(fill=tk.BOTH, expand=True)

    def display_suggestions(self, suggestions):
        for widget in self.suggestions_frame.winfo_children(): widget.destroy()
        canvas = tk.Canvas(self.suggestions_frame, bg='white', highlightthickness=0); v_scrollbar = ttk.Scrollbar(self.suggestions_frame, orient="vertical", command=canvas.yview); scrollable_frame = tk.Frame(canvas, bg='white'); scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))); canvas.create_window((0, 0), window=scrollable_frame, anchor="nw"); canvas.configure(yscrollcommand=v_scrollbar.set); v_scrollbar.pack(side="right", fill="y"); canvas.pack(side="left", fill="both", expand=True)
        for i, (title, item) in enumerate(suggestions.items()):
            bg = 'white' if i % 2 == 0 else '#fce4f0'
            tk.Label(scrollable_frame, text=f"• {title}:", font=('Arial', 10, 'bold'), bg=bg, fg=COLORS['Deep_Indigo'], anchor='nw').grid(row=i*2, sticky='w', padx=5, pady=2)
            tk.Label(scrollable_frame, text=item.get('text','N/A'), wraplength=270, justify=tk.LEFT, bg=bg).grid(row=i*2+1, sticky='w', padx=15, pady=2)
    
    def move_accessory(self, axis, delta):
        if not self.selected_accessory_name: return
        overrides = self.accessory_overrides.setdefault(self.selected_accessory_name, {'x_offset': 0, 'y_offset': 0, 'scale_factor': 1.0, 'rotation_offset': 0.0})
        overrides[f'{axis}_offset'] = overrides.get(f'{axis}_offset', 0) + delta
        self.redraw_if_static()

    def change_accessory_scale(self, delta):
        if not self.selected_accessory_name: return
        overrides = self.accessory_overrides.setdefault(self.selected_accessory_name, {'x_offset': 0, 'y_offset': 0, 'scale_factor': 1.0, 'rotation_offset': 0.0})
        overrides['scale_factor'] = max(0.1, min(3.0, overrides.get('scale_factor', 1.0) + delta))
        self.redraw_if_static()

    def change_accessory_rotation(self, delta):
        if not self.selected_accessory_name: return
        overrides = self.accessory_overrides.setdefault(self.selected_accessory_name, {'x_offset': 0, 'y_offset': 0, 'scale_factor': 1.0, 'rotation_offset': 0.0})
        overrides['rotation_offset'] = overrides.get('rotation_offset', 0) + delta
        self.redraw_if_static()

    def clear_accessory_override(self):
        if self.selected_accessory_name in self.accessory_overrides:
            self.accessory_overrides[self.selected_accessory_name] = {'x_offset': 0, 'y_offset': 0, 'scale_factor': 1.0, 'rotation_offset': 0.0}
            self.redraw_if_static()
            
    def redraw_if_static(self):
        if self.image_input_cv2 is not None and self.notebook.tab(self.notebook.select(), "text") == '🖼️ Upload Photo':
            self.display_static_image(self.image_input_cv2.copy())

    def start_webcam(self):
        global CAP, LIVE_STREAMING
        if not LIVE_STREAMING: CAP = cv2.VideoCapture(0); LIVE_STREAMING = True; self.landmark_history.clear(); self.start_face_scan(); self.process_webcam_frame()

    def stop_webcam(self):
        global CAP, LIVE_STREAMING
        if LIVE_STREAMING and CAP is not None: CAP.release(); CAP = None
        LIVE_STREAMING = False; self.is_scanning = False

    def load_image_file(self):
        self.stop_webcam()
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path: self.image_input_cv2 = cv2.imread(file_path); self.display_static_image(self.image_input_cv2.copy())

    def load_selected_accessories(self, event=None):
        self.selected_accessories = {}
        selected_names = [self.accessory_listbox.get(i) for i in self.accessory_listbox.curselection()]
        for name in selected_names:
            try:
                with open(os.path.join(ACCESORY_DIR, name), 'rb') as f: img_data = f.read()
                processed = process_accessory_image(img_data)
                if processed is not None:
                    self.selected_accessories[name] = processed; self.accessory_overrides.setdefault(name, {'x_offset': 0, 'y_offset': 0, 'scale_factor': 1.0, 'rotation_offset': 0.0})
            except Exception as e: print(f"Error loading {name}: {e}")
        if selected_names: self.selected_accessory_name = selected_names[-1]
        else: self.selected_accessory_name = None
        self.redraw_if_static()
    
    def display_static_image(self, frame):
        frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        landmarks = detect_and_get_landmarks(frame, use_dlib_detector=True)
        if landmarks:
            self.current_face_shape.set(classify_face_shape(landmarks)); self.update_suggestions()
            for name, img_data in self.selected_accessories.items():
                frame = overlay_accessory(frame, img_data, landmarks, determine_placement(name), self.accessory_overrides.get(name))
        else: self.current_face_shape.set("Face Not Found")
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.static_image_label.imgtk = imgtk; self.static_image_label.configure(image=imgtk)
        
    def save_favorite_look(self):
        frame_to_save, source = (self.last_frame_processed, "Webcam") if LIVE_STREAMING else (None, None)
        if not frame_to_save and self.image_input_cv2 is not None:
            source = "Upload"
            frame_to_save = cv2.resize(self.image_input_cv2.copy(), (VIDEO_WIDTH, VIDEO_HEIGHT))
            landmarks = detect_and_get_landmarks(frame_to_save, use_dlib_detector=True)
            if landmarks:
                for name, img_data in self.selected_accessories.items():
                    frame_to_save = overlay_accessory(frame_to_save, img_data, landmarks, determine_placement(name), self.accessory_overrides.get(name))
        if frame_to_save is None: messagebox.showinfo("Save Failed", "No active image to save."); return
        save_dir = "Favorites"; os.makedirs(save_dir, exist_ok=True)
        filename = f"Look_{time.strftime('%Y%m%d_%H%M%S')}_{self.current_face_shape.get()}.png"
        cv2.imwrite(os.path.join(save_dir, filename), frame_to_save)
        messagebox.showinfo("Save Successful", f"Look saved to 'Favorites' folder as {filename}")

if __name__ == "__main__":
    app = FaceFitApp()
    app.mainloop()