import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import mediapipe as mp
import numpy as np
import os
import json
import glob
from PIL import Image, ImageTk
import math
import threading
import time
from collections import Counter, defaultdict
import google.generativeai as genai
from dotenv import load_dotenv
import pickle
import traceback

# Local imports (user-provided)
from face_feature_extractor import landmarks_to_feature_vector
from trainmodel import MODEL_OUTPUT_PATH

# ---------------------------
# Config & Constants
# ---------------------------
APP_TITLE = "FaceFit Ultra - Smart AI Stylist"
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480
ACCESSORY_DIR = "accessories"
HAIRSTYLE_DIR = "hairstyles_dataset"
AI_REPORTS_CACHE = "ai_reports_cache.json"
GEMINI_CACHE = "gemini_cache.json"  # keep existing cache too
ACCESSORY_KEYWORDS = {
    'Glasses / Sunglasses': ['glass', 'sun', 'spectacle', 'eye', 'frame'],
    'Hats / Headwear': ['cap', 'hat', 'fedora', 'beanie', 'headband', 'crown', 'brim'],
    'Earrings / Jewelry': ['earring', 'hoop', 'stud', 'jewel', 'pendant', 'drop'],
    'Necklaces / Pendants': ['necklace', 'chain', 'pendant', 'scarf', 'tie', 'choker']
}

# Curated hairstyle-related tokens (used to extract keywords from AI text)
EXTRACT_TERMS = [
    "bob","pixie","layer","layers","wavy","wave","straight","curly","curl",
    "bang","bangs","fringe","fade","crop","pompadour","quiff","bun","lob",
    "side part","center part","waves","texture","slick","short","long","taper",
    "undercut","braid","braids","updo"
]

# ---------------------------
# Load environment
# ---------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ---------------------------
# Mediapipe init
# ---------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# ---------------------------
# Try to load ML model
# ---------------------------
_ml_model = None
def _load_ml_model():
    global _ml_model
    try:
        if os.path.exists(MODEL_OUTPUT_PATH):
            with open(MODEL_OUTPUT_PATH, "rb") as f:
                _ml_model = pickle.load(f)
            print(f"[INFO] Loaded ML model from {MODEL_OUTPUT_PATH}")
        else:
            print(f"[INFO] No ML model found at {MODEL_OUTPUT_PATH}. Using geometric fallback.")
    except Exception as e:
        _ml_model = None
        print("[WARN] Failed to load ML model:", e)
_load_ml_model()

# ---------------------------
# Utility functions
# ---------------------------
def safe_json_load(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def safe_json_save(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print("Failed to save json:", e)

def organize_accessories(ac_dir=ACCESSORY_DIR):
    cats = {k: [] for k in ACCESSORY_KEYWORDS.keys()}
    cats['Others'] = []
    if os.path.exists(ac_dir):
        for fn in sorted(os.listdir(ac_dir)):
            if not fn.lower().endswith(".png"):
                continue
            assigned = False
            for cat, kw in ACCESSORY_KEYWORDS.items():
                if any(k in fn.lower() for k in kw):
                    cats[cat].append(fn)
                    assigned = True
                    break
            if not assigned:
                cats['Others'].append(fn)
    return {k: v for k, v in cats.items() if v}

def load_hairstyle_images():
    styles = {}
    if not os.path.exists(HAIRSTYLE_DIR):
        return styles
    for g in ["Male","Female","Other"]:
        styles[g] = {}
        for s in ["Round","Square","Heart","Oval"]:
            folder = os.path.join(HAIRSTYLE_DIR, g, s)
            if os.path.exists(folder):
                styles[g][s] = glob.glob(os.path.join(folder, "*.*"))
            else:
                styles[g][s] = []
    return styles

ACCESSORY_CATEGORIES = organize_accessories()
HAIRSTYLES = load_hairstyle_images()

def process_accessory_image(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    # ensure has alpha for blending
    if img.shape[2] == 3:
        b,g,r = cv2.split(img)
        alpha = np.full(b.shape, 255, dtype=b.dtype)
        img = cv2.merge((b,g,r,alpha))
    return img

# ---------------------------
# Geometric shape detection (unchanged logic, minor safety)
# ---------------------------
def get_face_shape_geometric(landmarks, frame_w, frame_h):
    try:
        p_jaw_l = landmarks[172]; p_jaw_r = landmarks[397]
        p_cheek_l = landmarks[234]; p_cheek_r = landmarks[454]
        p_forehead = landmarks[10]; p_chin = landmarks[152]
        face_width_jaw = math.hypot((p_jaw_l.x - p_jaw_r.x)*frame_w, (p_jaw_l.y - p_jaw_r.y)*frame_h)
        face_width_cheeks = math.hypot((p_cheek_l.x - p_cheek_r.x)*frame_w, (p_cheek_l.y - p_cheek_r.y)*frame_h)
        face_height = math.hypot((p_forehead.x - p_chin.x)*frame_w, (p_forehead.y - p_chin.y)*frame_h)
        face_shape = "Oval"
        if face_width_jaw > 0 and face_height > 0:
            if abs(face_width_cheeks - face_height) < 25:
                face_shape = "Round"
            elif abs(face_width_jaw - face_width_cheeks) < 30:
                face_shape = "Square"
            elif face_width_cheeks > face_width_jaw and (face_height / face_width_cheeks) < 1.3:
                face_shape = "Heart"
        return face_shape
    except Exception:
        return "Oval"

# ---------------------------
# Enhanced combined detection
# ---------------------------
def get_face_shape_combined(landmarks, frame_w, frame_h):
    # First try ML model (if available)
    global _ml_model
    geo = get_face_shape_geometric(landmarks, frame_w, frame_h)
    if _ml_model is not None:
        try:
            # convert mediapipe landmark objects to tuples (x, y)
            pts = [(lm.x * frame_w, lm.y * frame_h) for lm in landmarks]
            features = landmarks_to_feature_vector(pts)
            if features.size > 0:
                pred = _ml_model.predict([features])[0]
                # If model supports predict_proba, check confidence
                try:
                    proba = max(_ml_model.predict_proba([features])[0])
                    # If high confidence, prefer ML; else fallback to geometric
                    if proba >= 0.6:
                        return pred
                    else:
                        # use geometric if probabilities are low
                        return geo
                except Exception:
                    return pred
        except Exception as e:
            print("ML prediction error:", e)
            return geo
    return geo

# ---------------------------
# Mediapipe detection helper
# ---------------------------
def detect_with_mediapipe(img):
    res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if res and res.multi_face_landmarks:
        return res.multi_face_landmarks[0].landmark
    return None

# ---------------------------
# Overlay helpers (kept as in original)
# ---------------------------
def _blend_transparent(bg, ov, x, y):
    try:
        h,w,_ = bg.shape
        oh,ow,_ = ov.shape
        x1,y1 = max(x,0), max(y,0)
        x2,y2 = min(x+ow, w), min(y+oh, h)
        roi = bg[y1:y2, x1:x2]
        if roi.shape[0] == 0 or roi.shape[1] == 0:
            return bg
        ov_x1, ov_y1 = max(0,-x), max(0,-y)
        ov_x2, ov_y2 = ov_x1 + (x2-x1), ov_y1 + (y2-y1)
        ov_p = ov[ov_y1:ov_y2, ov_x1:ov_x2]
        mask = (ov_p[:,:,3]/255.0)[:,:,np.newaxis]
        roi_b = (1.0-mask) * roi.astype(float) + mask * ov_p[:,:,:3].astype(float)
        bg[y1:y2, x1:x2] = roi_b.astype(np.uint8)
    except Exception:
        pass
    return bg

def get_landmarks_pixels(landmarks,w,h):
    return {i: (lm.x*w, lm.y*h) for i,lm in enumerate(landmarks)}

def get_angle_and_width(p1,p2):
    dx,dy = p2[0]-p1[0], p2[1]-p1[1]
    return math.degrees(math.atan2(dy,dx)), math.hypot(dx,dy)

def overlay_glasses(bg,ov,lms,ovr):
    _, hw = get_angle_and_width(lms[234], lms[454])
    a, _ = get_angle_and_width(lms[33], lms[263])
    ey = (lms[33][1] + lms[263][1]) / 2.0
    cx, cy = int(lms[168][0]), int(ey)
    tw = int(hw * 0.95 * ovr.get('scale_factor',1.0))
    th = int(tw * (ov.shape[0]/ov.shape[1])) if ov.shape[1] > 0 else 0
    if tw == 0 or th == 0: return bg
    r = cv2.resize(ov, (tw, th))
    M = cv2.getRotationMatrix2D((tw/2, th/2), a, 1)
    rot = cv2.warpAffine(r, M, (tw, th))
    return _blend_transparent(bg, rot, cx - tw//2, cy - th//2)

def overlay_hat(bg,ov,lms,ovr):
    _, hw = get_angle_and_width(lms[234], lms[454])
    a, _ = get_angle_and_width(lms[33], lms[263])
    cx, cy = int(lms[10][0]), int(lms[10][1])
    tw = int(hw * 1.4 * ovr.get('scale_factor',1.0))
    th = int(tw * (ov.shape[0]/ov.shape[1])) if ov.shape[1] > 0 else 0
    if tw==0 or th==0: return bg
    r = cv2.resize(ov, (tw, th))
    M = cv2.getRotationMatrix2D((tw/2, th/2), a, 1)
    rot = cv2.warpAffine(r, M, (tw, th))
    return _blend_transparent(bg, rot, cx - tw//2, cy - int(th*0.8))

def overlay_earrings(bg,ov,lms,ovr):
    _, hw = get_angle_and_width(lms[234], lms[454])
    tw = int(hw * 0.15 * ovr.get('scale_factor',1.0))
    th = int(tw * (ov.shape[0]/ov.shape[1])) if ov.shape[1] > 0 else 0
    if tw==0 or th==0: return bg
    r = cv2.resize(ov, (tw, th))
    bg = _blend_transparent(bg, r, int(lms[132][0] - tw/2), int(lms[132][1] - th/2))
    return _blend_transparent(bg, r, int(lms[361][0] - tw/2), int(lms[361][1] - th/2))

def overlay_necklace(bg,ov,lms,ovr):
    a, jw = get_angle_and_width(lms[130], lms[359])
    cx, cy = int(lms[152][0]), int(lms[152][1])
    tw = int(jw * 1.5 * ovr.get('scale_factor',1.0))
    th = int(tw * (ov.shape[0]/ov.shape[1])) if ov.shape[1] > 0 else 0
    if tw==0 or th==0: return bg
    r = cv2.resize(ov, (tw, th))
    M = cv2.getRotationMatrix2D((tw/2, th/2), a, 1)
    rot = cv2.warpAffine(r, M, (tw, th))
    return _blend_transparent(bg, rot, cx - tw//2, cy + int(th*0.1))

# ---------------------------
# Gemini / AI report integration
# ---------------------------
def get_gemini_vision_report(api_key, image, face_shape, gender, extra_prompt=""):
    """
    Wrap Gemini call. image should be a PIL image object or bytes accepted by model.
    Returns text (string) or error message.
    """
    try:
        if not api_key:
            raise RuntimeError("No API key")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        prompt = f"""Act as a world-class personal stylist writing an exclusive style note for a client. Your tone is chic, insightful, and warmly personal, like a message from a trusted friend with an amazing eye for fashion. Avoid generic compliments and AI-sounding phrases. Here's the client's photo. I've determined their face shape is **{face_shape}** and they identify as **{gender}**. Based on the image, write a style guide for them. Instead of a rigid report, structure your response as a friendly, flowing message. Start with their overall vibe. Comment on a feature you notice in the photo and connect it to their {face_shape} face shape in a positive, authentic way. Next, talk about eyewear. Suggest two distinct styles. Name the style and explain specifically how it complements their features. Then, move to hairstyles. Recommend two different cuts or styles that would be flattering. Be descriptive. Finally, suggest a color palette of 4-5 complementary colors. Conclude with a warm sign-off. {extra_prompt}"""
        # the model accepts images as well; pass the PIL image directly if possible
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"An error occurred while generating the AI report.\n\nError: {e}"

# ---------------------------
# Keyword extraction helpers
# ---------------------------
def extract_style_keywords(text):
    text_l = text.lower()
    found = []
    for term in EXTRACT_TERMS:
        if term in text_l:
            found.append(term)
    # normalize plural/similar terms (simple)
    unique = []
    for t in found:
        base = t.rstrip('s')
        if base not in unique:
            unique.append(base)
    return unique

def extract_color_keywords(text):
    # naive color extraction — pick common colors
    colors = ["black","white","beige","brown","blonde","blond","blondish","auburn","red","ginger","silver","gray","grey","blonde","brunette","gold","navy","green","blue","teal","salmon","peach","ivory"]
    res = []
    tl = text.lower()
    for c in colors:
        if c in tl:
            res.append(c)
    return list(dict.fromkeys(res))

def extract_accessory_keywords(text):
    res = []
    tl = text.lower()
    for cat, kwlist in ACCESSORY_KEYWORDS.items():
        for kw in kwlist:
            if kw in tl:
                res.append(cat)
                break
    return list(dict.fromkeys(res))

# ---------------------------
# GUI App (optimized and threaded)
# ---------------------------
class WebcamCaptureThread(threading.Thread):
    def __init__(self, src=0):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
        self.frame = None
        self.running = False
        self.lock = threading.Lock()

    def run(self):
        self.running = True
        while self.running:
            ret, fr = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = fr
        self.cap.release()

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False

class FaceFitUltraApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1100x760")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # state
        self.video_stream = None
        self.live_streaming = False
        self.last_frame_raw = None
        self.last_frame_composed = None
        self.last_landmarks = None

        self.selected_accessories = {}
        self.accessory_overrides = {}
        self.selected_accessory_name = None

        self.gender_var = tk.StringVar(value="Female")
        self.hair_gender_var = tk.StringVar(value="Female")
        self.hair_shape_var = tk.StringVar(value="Oval")
        self.personality_var = tk.StringVar(value="")  # user-supplied vibe
        self.CACHE_FILE = GEMINI_CACHE
        self.AI_REPORTS_FILE = AI_REPORTS_CACHE

        self.hairstyles = HAIRSTYLES
        self.accessory_categories = ACCESSORY_CATEGORIES

        # smart keywords from last AI run
        self.smart_keywords = []
        self.smart_colors = []
        self.smart_accessory_cats = []

        self._tooltip = None

        self._build_ui()
        # load accessory list after UI ready
        self.after(60, self._load_accessories_list)

    def _build_ui(self):
        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=3)
        main.add(right, weight=1)

        # Left: video / upload tabs
        self.left_nb = ttk.Notebook(left)
        self.left_nb.pack(fill=tk.BOTH, expand=True)
        cam_frame = ttk.Frame(self.left_nb)
        upload_frame = ttk.Frame(self.left_nb)
        self.left_nb.add(cam_frame, text="📹 Live")
        self.left_nb.add(upload_frame, text="🖼️ Upload")

        # video display
        self.vid_lbl = ttk.Label(cam_frame)
        self.vid_lbl.pack(fill=tk.BOTH, expand=True)

        # upload controls
        up_ctrl = ttk.Frame(upload_frame)
        up_ctrl.pack(fill=tk.X, padx=6, pady=6)
        ttk.Button(up_ctrl, text="Load Photo", command=self.load_image_file).pack(side=tk.LEFT)
        self.static_lbl = ttk.Label(upload_frame)
        self.static_lbl.pack(fill=tk.BOTH, expand=True)

        # Right: controls and tabs
        ctrl_fr = ttk.LabelFrame(right, text="Controls")
        ctrl_fr.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(ctrl_fr, text="Start Cam", command=self.start_webcam).grid(row=0, column=0, sticky='ew', padx=2)
        ttk.Button(ctrl_fr, text="Stop Cam", command=self.stop_webcam).grid(row=0, column=1, sticky='ew', padx=2)
        ttk.Button(ctrl_fr, text="Save Look", command=self.save_favorite_look).grid(row=1, column=0, columnspan=2, sticky='ew', padx=2)

        # Extra: small input for user 'vibe'
        vibe_fr = ttk.Frame(ctrl_fr)
        vibe_fr.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(6,0))
        ttk.Label(vibe_fr, text="Your vibe (optional):").pack(side=tk.LEFT, padx=(2,6))
        ttk.Entry(vibe_fr, textvariable=self.personality_var, width=18).pack(side=tk.LEFT)

        # Right: tabs for AI profile and try-on and hairstyles
        self.right_nb = ttk.Notebook(right)
        self.right_nb.pack(fill=tk.BOTH, expand=True, pady=6)
        self.tab_ai = ttk.Frame(self.right_nb)
        self.tab_tryon = ttk.Frame(self.right_nb)
        self.tab_hair = ttk.Frame(self.right_nb)
        self.right_nb.add(self.tab_ai, text="⭐ AI Profile")
        self.right_nb.add(self.tab_tryon, text="👓 Try-On")
        self.right_nb.add(self.tab_hair, text="💇 Hairstyles")

        # AI tab content
        prof_fr = ttk.LabelFrame(self.tab_ai, text="1. Your Profile")
        prof_fr.pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(prof_fr, text="Gender:").grid(row=0, column=0, sticky='w', padx=5)
        ttk.Combobox(prof_fr, textvariable=self.gender_var, values=['Female','Male','Other'], state='readonly').grid(row=0, column=1, sticky='ew', padx=5)

        ttk.Button(self.tab_ai, text="2. Generate AI Style Profile", command=self.run_gemini_analysis, style="Accent.TButton").pack(fill=tk.X, padx=6, pady=(4,8))
        # Re-Analyze button
        ttk.Button(self.tab_ai, text="🧠 Smart Re-Analyze", command=self.reanalyze_current, style="Accent.TButton").pack(fill=tk.X, padx=6, pady=(0,8))

        # AI report area
        self.rep_frame = ttk.LabelFrame(self.tab_ai, text="3. Your AI Style Guide")
        self.rep_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.rep_txt = tk.Text(self.rep_frame, wrap=tk.WORD, height=12)
        self.rep_txt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.rep_txt.insert(tk.END, "Click 'Generate AI Style Profile' to get started.")

        # Button container under AI report (View Hairstyles / Export)
        self.view_hair_btn_container = ttk.Frame(self.rep_frame)
        self.view_hair_btn_container.pack(fill=tk.X, padx=6, pady=(2,6))

        # Try-on tab content
        acc_fr = ttk.LabelFrame(self.tab_tryon, text="Select Accessory")
        acc_fr.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.cat_var = tk.StringVar()
        self.cat_cb = ttk.Combobox(acc_fr, textvariable=self.cat_var, state='readonly')
        self.cat_cb.pack(fill=tk.X, padx=6, pady=4)
        self.cat_cb.bind('<<ComboboxSelected>>', self._load_accessories_by_category)
        self.acc_lb = tk.Listbox(acc_fr, height=8, exportselection=False)
        self.acc_lb.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.acc_lb.bind('<<ListboxSelect>>', self._on_accessory_select)

        # hairstyle tab content
        hair_ctrl_fr = ttk.LabelFrame(self.tab_hair, text="Find Your Style")
        hair_ctrl_fr.pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(hair_ctrl_fr, text="Gender:").grid(row=0, column=0, padx=5, pady=2, sticky='w')
        ttk.Combobox(hair_ctrl_fr, textvariable=self.hair_gender_var, values=['Female','Male','Other'], state='readonly').grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(hair_ctrl_fr, text="Face Shape:").grid(row=1, column=0, padx=5, pady=2, sticky='w')
        ttk.Combobox(hair_ctrl_fr, textvariable=self.hair_shape_var, values=['Oval','Round','Square','Heart'], state='readonly').grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(hair_ctrl_fr, text="Find Hairstyles", command=self._manual_hairstyle_search).grid(row=2, column=0, columnspan=2, sticky='ew', padx=5, pady=6)

        hair_gallery_fr = ttk.LabelFrame(self.tab_hair, text="Suggestions")
        hair_gallery_fr.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.hair_canvas = tk.Canvas(hair_gallery_fr)
        self.hair_scrollbar = ttk.Scrollbar(hair_gallery_fr, orient="vertical", command=self.hair_canvas.yview)
        self.scrollable_hair_frame = ttk.Frame(self.hair_canvas)
        self.scrollable_hair_frame.bind("<Configure>", lambda e: self.hair_canvas.configure(scrollregion=self.hair_canvas.bbox("all")))
        self.hair_canvas.create_window((0,0), window=self.scrollable_hair_frame, anchor="nw")
        self.hair_canvas.configure(yscrollcommand=self.hair_scrollbar.set)
        self.hair_canvas.pack(side="left", fill="both", expand=True)
        self.hair_scrollbar.pack(side="right", fill="y")

        ttk.Style().configure("Accent.TButton", font=('Helvetica', 10, 'bold'), foreground='green')

        # keyboard bindings
        self.bind('<plus>', lambda e: self._scale_selected(0.05))
        self.bind('<minus>', lambda e: self._scale_selected(-0.05))

    # ---------------------------
    # Webcam, image loading, UI update loops
    # ---------------------------
    def start_webcam(self):
        if not self.live_streaming:
            self.video_stream = WebcamCaptureThread()
            self.video_stream.start()
            self.live_streaming = True
            self.left_nb.select(0)
            self._update_webcam_loop()

    def stop_webcam(self):
        if self.live_streaming:
            try:
                self.video_stream.stop()
            except Exception:
                pass
            self.live_streaming = False

    def _update_webcam_loop(self):
        if not self.live_streaming:
            return
        fr = self.video_stream.read()
        if fr is None:
            self.after(10, self._update_webcam_loop)
            return
        fr = cv2.flip(fr, 1)
        self.last_frame_raw = fr
        comp = self.process_frame_for_tryon(fr)
        self.last_frame_composed = comp
        imgtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)))
        self.vid_lbl.imgtk = imgtk
        self.vid_lbl.configure(image=imgtk)
        self.after(30, self._update_webcam_loop)

    def load_image_file(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if not path:
            return
        self.stop_webcam()
        fr = cv2.imread(path)
        if fr is None:
            messagebox.showerror("Error", "Could not load image.")
            return
        h,w,_ = fr.shape
        sc = min(VIDEO_WIDTH/w, VIDEO_HEIGHT/h) if min(w,h) > 1 else 1
        fr = cv2.resize(fr, (int(w*sc), int(h*sc)))
        self.last_frame_raw = fr
        self.refresh_static_image()
        self.left_nb.select(1)

    def _on_accessory_select(self, event=None):
        sel = [self.acc_lb.get(i) for i in self.acc_lb.curselection()]
        self.selected_accessories.clear()
        for name in sel:
            try:
                with open(os.path.join(ACCESSORY_DIR, name), "rb") as fh:
                    p = process_accessory_image(fh.read())
                    if p is not None:
                        self.selected_accessories[name] = p
                        self.accessory_overrides.setdefault(name, {'scale_factor': 1.0})
            except Exception as e:
                print("Error loading accessory", e)
        self.selected_accessory_name = sel[-1] if sel else None
        if self.left_nb.index(self.left_nb.select()) == 1 and self.last_frame_raw is not None:
            self.refresh_static_image()

    def refresh_static_image(self):
        if self.last_frame_raw is None:
            return
        comp = self.process_frame_for_tryon(self.last_frame_raw)
        self.last_frame_composed = comp
        imgtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)))
        self.static_lbl.imgtk = imgtk
        self.static_lbl.configure(image=imgtk)

    def process_frame_for_tryon(self, fr):
        self.last_landmarks = detect_with_mediapipe(fr)
        comp = fr.copy()
        if self.last_landmarks:
            lms_px = get_landmarks_pixels(self.last_landmarks, fr.shape[1], fr.shape[0])
            for name, img in self.selected_accessories.items():
                if img is not None:
                    ovr = self.accessory_overrides.get(name, {})
                    p = self.get_placement_from_name(name)
                    if p == 'glasses':
                        comp = overlay_glasses(comp, img, lms_px, ovr)
                    elif p == 'hat':
                        comp = overlay_hat(comp, img, lms_px, ovr)
                    elif p == 'earrings':
                        comp = overlay_earrings(comp, img, lms_px, ovr)
                    elif p == 'necklace':
                        comp = overlay_necklace(comp, img, lms_px, ovr)
        return comp

    def get_placement_from_name(self, name):
        nl = name.lower()
        for cat, kw in ACCESSORY_KEYWORDS.items():
            if any(k in nl for k in kw):
                if cat == 'Glasses / Sunglasses': return 'glasses'
                if cat == 'Hats / Headwear': return 'hat'
                if cat == 'Earrings / Jewelry': return 'earrings'
                if cat == 'Necklaces / Pendants': return 'necklace'
        return 'default'

    # ---------------------------
    # AI Profile generation & caching
    # ---------------------------
    def run_gemini_analysis(self):
        # Validate image available
        if self.left_nb.index(self.left_nb.select()) == 0 and self.live_streaming:
            fr = self.last_frame_raw
        else:
            fr = self.last_frame_raw
        if fr is None:
            messagebox.showerror("Error", "No image to analyze.")
            return

        lms = detect_with_mediapipe(fr)
        if not lms:
            messagebox.showerror("Error", "Could not detect a face.")
            return

        h,w,_ = fr.shape
        face_shape = get_face_shape_combined(lms, w, h)

        # show generating message
        self.rep_txt.delete(1.0, tk.END)
        self.rep_txt.insert(tk.END, "🚀 Generating your AI Style Profile...")
        self.update_idletasks()

        # set hairstyle controls but do not auto-switch
        self.hair_gender_var.set(self.gender_var.get())
        self.hair_shape_var.set(face_shape)

        # show "View Hairstyles" button (refresh container)
        for widget in self.view_hair_btn_container.winfo_children():
            widget.destroy()
        def open_hair_tab():
            self._update_hairstyle_suggestions()
            self.right_nb.select(self.tab_hair)
        btn = ttk.Button(self.view_hair_btn_container, text="💇 View Suggested Hairstyles", command=open_hair_tab, style="Accent.TButton")
        btn.pack(fill=tk.X)

        # prepare PIL image for sending to Gemini
        try:
            pil_img = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        except Exception:
            pil_img = None

        # Submit to background worker
        threading.Thread(target=self._generate_and_display_report, args=(pil_img, face_shape, self.gender_var.get()), daemon=True).start()

    def _generate_and_display_report(self, pil_img, face_shape, gender):
        """
        Worker thread: call Gemini (if API key) or fallback local report,
        extract keywords, store in ai_reports_cache.json and update UI.
        """
        cache = safe_json_load(self.AI_REPORTS_FILE)
        cache_key = f"{face_shape.lower()}_{gender.lower()}"
        # optional personality prompt
        personality = self.personality_var.get().strip()
        extra = ""
        if personality:
            extra = f"Also, the user describes their vibe as: '{personality}'. Tailor recommendations to that vibe."

        # If cached and exists, use it
        if cache_key in cache:
            rep_text = cache[cache_key].get("report_text", "")
            keywords = cache[cache_key].get("keywords", [])
            colors = cache[cache_key].get("colors", [])
            accessory_cats = cache[cache_key].get("accessory_cats", [])
        else:
            if GEMINI_API_KEY:
                try:
                    rep_text = get_gemini_vision_report(GEMINI_API_KEY, pil_img, face_shape, gender, extra_prompt=extra)
                except Exception as e:
                    rep_text = f"AI generation error: {e}\n\nShowing local suggestions."
            else:
                rep_text = self.get_local_style_report(face_shape, gender)
            keywords = extract_style_keywords(rep_text)
            colors = extract_color_keywords(rep_text)
            accessory_cats = extract_accessory_keywords(rep_text)
            # Save to cache
            cache[cache_key] = {
                "report_text": rep_text,
                "keywords": keywords,
                "colors": colors,
                "accessory_cats": accessory_cats,
                "timestamp": time.time()
            }
            safe_json_save(self.AI_REPORTS_FILE, cache)

        # update UI in main thread
        def ui_update():
            self.rep_txt.delete(1.0, tk.END)
            self.rep_txt.insert(tk.END, rep_text)
            self.smart_keywords = keywords
            self.smart_colors = colors
            self.smart_accessory_cats = accessory_cats
            # small visual feedback
            if keywords:
                self.rep_txt.insert(tk.END, "\n\nDetected hairstyle keywords: " + ", ".join(keywords))
            if colors:
                self.rep_txt.insert(tk.END, "\nDetected color suggestions: " + ", ".join(colors))
        self.after(10, ui_update)

    def reanalyze_current(self):
        """
        Re-run Gemini/local with the same last image and updated personality or gender selection.
        """
        if self.last_frame_raw is None:
            messagebox.showerror("Error", "No image loaded to re-analyze.")
            return
        lms = detect_with_mediapipe(self.last_frame_raw)
        if not lms:
            messagebox.showerror("Error", "Could not detect a face.")
            return
        h,w,_ = self.last_frame_raw.shape
        face_shape = get_face_shape_combined(lms, w, h)
        try:
            pil_img = Image.fromarray(cv2.cvtColor(self.last_frame_raw, cv2.COLOR_BGR2RGB))
        except Exception:
            pil_img = None
        threading.Thread(target=self._generate_and_display_report, args=(pil_img, face_shape, self.gender_var.get()), daemon=True).start()

    # ---------------------------
    # Hairstyle gallery + smart highlighting
    # ---------------------------
    def _manual_hairstyle_search(self):
        self._update_hairstyle_suggestions()

    def _update_hairstyle_suggestions(self):
        # improved gallery: highlight items matching self.smart_keywords
        gender = self.hair_gender_var.get()
        face_shape = self.hair_shape_var.get()
        for w in self.scrollable_hair_frame.winfo_children():
            w.destroy()

        dataset = self.hairstyles.get(gender, {}).get(face_shape, [])
        if not dataset:
            ttk.Label(self.scrollable_hair_frame, text="No hairstyle images found for this selection.").pack(padx=10, pady=10)
            return

        # compute matching score: if any smart keyword in filename or path
        def match_score(path):
            name = os.path.basename(path).lower()
            score = 0
            for kw in self.smart_keywords:
                if kw in name:
                    score += 2
            for kw in self.smart_colors:
                if kw in name:
                    score += 1
            return score

        scored = [(match_score(p), p) for p in dataset]
        # sort by score desc
        scored.sort(key=lambda x: (-x[0], x[1]))

        cols = 3
        thumb = 160
        for idx, (_, img_path) in enumerate(scored):
            try:
                pil_img = Image.open(img_path).resize((thumb, thumb), Image.LANCZOS)
                tk_img = ImageTk.PhotoImage(pil_img)
                name = os.path.basename(img_path).split('.')[0].replace('_',' ').replace('-',' ').title()
                score = match_score(img_path)
                border = "#27ae60" if score > 0 else "#cccccc"
                fr = tk.Frame(self.scrollable_hair_frame, highlightbackground=border, highlightthickness=3, bd=0)
                fr.grid(row=idx // cols, column=idx % cols, padx=8, pady=8)
                lbl = ttk.Label(fr, image=tk_img, cursor="hand2")
                lbl.image = tk_img
                lbl.pack()
                ttl = ttk.Label(fr, text=name, wraplength=thumb, justify='center')
                ttl.pack(fill=tk.X, pady=(4,0))
                lbl.bind("<Button-1>", lambda e, p=img_path: self._show_full_hairstyle(p))
                lbl.bind("<Enter>", lambda e, t=name: self._show_tooltip(e.widget, t))
                lbl.bind("<Leave>", lambda e: self._hide_tooltip())
            except Exception as e:
                print("Error loading hairstyle", img_path, e)

    # ---------------------------
    # tooltips
    # ---------------------------
    def _show_tooltip(self, widget, text):
        try:
            x = widget.winfo_rootx() + 20
            y = widget.winfo_rooty() + 20
            self._tooltip = tk.Toplevel(self)
            self._tooltip.wm_overrideredirect(True)
            self._tooltip.geometry(f"+{x}+{y}")
            lbl = tk.Label(self._tooltip, text=text, bg="#ffffe0", relief="solid", borderwidth=1, padx=4, pady=2)
            lbl.pack()
        except Exception:
            pass

    def _hide_tooltip(self):
        try:
            if self._tooltip:
                self._tooltip.destroy()
                self._tooltip = None
        except Exception:
            pass

    def _show_full_hairstyle(self, img_path):
        if not os.path.exists(img_path):
            return
        try:
            pil_img = Image.open(img_path).resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.LANCZOS)
            comp = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            self.last_frame_composed = comp
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)))
            target_label = self.vid_lbl if self.left_nb.index(self.left_nb.select()) == 0 else self.static_lbl
            target_label.imgtk = imgtk
            target_label.configure(image=imgtk)
        except Exception as e:
            messagebox.showerror("Error", f"Could not display hairstyle image.\n{e}")

    # ---------------------------
    # accessory helpers & list loading
    # ---------------------------
    def _load_accessories_list(self):
        cats = list(self.accessory_categories.keys())
        self.cat_cb.config(values=cats)
        if cats:
            self.cat_var.set(cats[0])
            self._load_accessories_by_category()

    def _load_accessories_by_category(self, event=None):
        cat = self.cat_var.get()
        self.acc_lb.delete(0, tk.END)
        for f in self.accessory_categories.get(cat, []):
            self.acc_lb.insert(tk.END, f)

    def _scale_selected(self, delta):
        if self.selected_accessory_name:
            ov = self.accessory_overrides.setdefault(self.selected_accessory_name, {'scale_factor': 1.0})
            ov['scale_factor'] = max(0.2, min(3.0, ov.get('scale_factor', 1.0) + delta))
            if self.left_nb.index(self.left_nb.select()) == 1 and self.last_frame_raw is not None:
                self.refresh_static_image()

    # ---------------------------
    # Save look and exit
    # ---------------------------
    def save_favorite_look(self):
        if self.last_frame_composed is not None:
            os.makedirs("Favorites", exist_ok=True)
            fname = f"Favorites/Look_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(fname, self.last_frame_composed)
            messagebox.showinfo("Saved", f"Saved to {fname}")
        else:
            messagebox.showinfo("Error", "No image to save.")

    def _on_close(self):
        try:
            self.stop_webcam()
        except Exception:
            pass
        self.destroy()

# ---------------------------
# Helper: local style guide
# ---------------------------
def local_style_report(face_shape, gender):
    report = f"### Local Style Guide for {gender} with {face_shape} Face ###\n\n"
    hairstyles = {"Round":["Layered Bob","Side-swept Bangs"], "Square":["Soft Waves","Long Layers"],
                  "Heart":["Lob (Long Bob)","Pixie Cut"], "Oval":["Blunt Bob","Almost any style works!"]}
    if gender == "Male":
        hairstyles = {"Round":["Textured Crop","Pompadour"], "Square":["Buzz Cut","Slicked Back"],
                      "Heart":["Side Part","Longer fringe"], "Oval":["Quiff","Almost any style"]}
    eyewear = {"Round":["Rectangular Frames","Wayfarers"], "Square":["Round Frames","Aviators"],
               "Heart":["Cat-Eye Glasses","Rimless Frames"], "Oval":["Most frames work","Geometric"]}
    report += "**Eyewear Recommendations:**\n"
    for item in eyewear.get(face_shape, []):
        report += f"- {item}\n"
    report += "\n**Hairstyle Suggestions:**\n"
    for item in hairstyles.get(face_shape, []):
        report += f"- {item}\n"
    report += "\n**Suggested Color Palette:**\n- Neutral Tones\n- Accent Colors\n"
    return report

# ---------------------------
# Run app
# ---------------------------
if __name__ == "__main__":
    # Ensure cache file exists
    if not os.path.exists(AI_REPORTS_CACHE):
        safe_json_save(AI_REPORTS_CACHE, {})
    if not os.path.exists(GEMINI_CACHE):
        safe_json_save(GEMINI_CACHE, {})

if __name__ == "__main__":
    app = FaceFitUltraApp()
    app.mainloop()


