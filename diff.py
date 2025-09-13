import cv2
import numpy as np
import dlib
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import os
import csv

# Make sure you have the 'shape_predictor_68_face_landmarks.dat' file in the same directory.
# Download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

class AIAccessorySuggestorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FaceFit AI Accessory Suggestor")
        self.root.geometry("400x500")
        self.root.resizable(False, False)

        self.detector = dlib.get_frontal_face_detector()
        try:
            self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            self.landmark_available = True
        except:
            self.landmark_available = False

        self.scaler = StandardScaler()
        self.knn_model = KNeighborsClassifier(n_neighbors=3)
        self.is_trained = False
        self.feature_db = []
        self.accessory_db = []
        self.training_data_file = 'training_data.csv'
        
        self.load_training_data()
        self.train_model_gui()
        
        self.accessory_images = self.load_accessory_images()

        self.create_main_menu()

    def load_accessory_images(self):
        """Returns a dictionary of available accessories with their image paths."""
        return {
            'Aviator glasses': {'type': 'glasses', 'path': 'accessories/aviator_glasses.png'},
            'Hoop earrings': {'type': 'earrings', 'path': 'accessories/hoop_earrings.png'},
            'Rectangular glasses': {'type': 'glasses', 'path': 'accessories/rectangular_glasses.png'},
            'Round glasses': {'type': 'glasses', 'path': 'accessories/round_glasses.png'},
            'Pearl necklace': {'type': 'necklace', 'path': 'accessories/pearl_necklace.png'},
            'Chain necklace': {'type': 'necklace', 'path': 'accessories/chain_necklace.png'},
            'Stud earrings': {'type': 'earrings', 'path': 'accessories/stud_earrings.png'},
        }

    def create_main_menu(self):
        """Builds the main menu window with all mode buttons."""
        frame = ttk.Frame(self.root, padding="20")
        frame.pack(expand=True, fill=tk.BOTH)

        label = ttk.Label(frame, text="FaceFit AI Accessory Suggestor", font=("Helvetica", 16))
        label.pack(pady=10)

        btn_train = ttk.Button(frame, text="Train Model", command=self.train_model_gui)
        btn_train.pack(pady=10, fill=tk.X)

        btn_start_training = ttk.Button(frame, text="Start Data Collection", command=self.open_training_window)
        btn_start_training.pack(pady=10, fill=tk.X)

        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=10)

        btn_analysis = ttk.Button(frame, text="Analysis Mode", command=lambda: self.open_mode_window('analysis'))
        btn_analysis.pack(pady=10, fill=tk.X)

        btn_suggestion = ttk.Button(frame, text="Suggestion Mode", command=lambda: self.open_mode_window('suggestion'))
        btn_suggestion.pack(pady=10, fill=tk.X)

        btn_try_on = ttk.Button(frame, text="Try-On Mode", command=lambda: self.open_mode_window('try-on'))
        btn_try_on.pack(pady=10, fill=tk.X)

        btn_exit = ttk.Button(frame, text="Exit", command=self.root.destroy)
        btn_exit.pack(pady=10, fill=tk.X)

    def open_training_window(self):
        """Opens a new window for training mode."""
        TrainingModeWindow(self.root, self)

    def open_mode_window(self, mode):
        """Opens a new window for the selected mode."""
        ModeWindow(self.root, self, mode=mode)

    def train_model_gui(self):
        """Trains the model and shows a confirmation message."""
        try:
            if not self.feature_db:
                messagebox.showwarning("No Data", "No training data available. Please collect some data first.")
                return False
            
            X = np.array(self.feature_db)
            y = self.accessory_db
            
            if X.shape[0] > 0 and X.shape[1] > 0:
                self.scaler.fit(X)
                X_scaled = self.scaler.transform(X)
                self.knn_model.fit(X_scaled, y)
                self.is_trained = True
                messagebox.showinfo("Training Complete", f"Model successfully trained with {len(y)} data points.")
                return True
            else:
                messagebox.showwarning("Training Failed", "Training data is not in the correct format.")
                return False
        except Exception as e:
            messagebox.showerror("Training Failed", f"An error occurred during training: {e}")
            return False

    def load_training_data(self):
        """Loads training data from a CSV file and adds initial sample data."""
        self.feature_db = []
        self.accessory_db = []
        
        self.add_sample_data()
        
        if os.path.exists(self.training_data_file):
            with open(self.training_data_file, mode='r', newline='') as file:
                reader = csv.reader(file)
                for row in reader:
                    features = [float(x) for x in row[:-1]]
                    accessory = row[-1]
                    self.feature_db.append(features)
                    self.accessory_db.append(accessory)

    def add_sample_data(self):
        """Adds a minimal hardcoded dataset for initial training."""
        sample_features = [
            [0, 0, 1, 0.7, 1.0], [0, 1, 2, 0.7, 1.0], [0, 2, 1, 0.7, 1.0],
            [1, 0, 0, 1.0, 1.0], [1, 1, 1, 1.0, 1.0], [1, 2, 2, 1.0, 1.0],
        ]
        sample_accessories = [
            'Aviator glasses', 'Round glasses', 'Cat-eye glasses',
            'Hoop earrings', 'Statement earrings', 'Stud earrings',
        ]
        self.feature_db.extend(sample_features)
        self.accessory_db.extend(sample_accessories)
    
    def save_data_point(self, features, accessory_name):
        """Saves a new data point to the CSV file and in memory."""
        self.feature_db.append(features)
        self.accessory_db.append(accessory_name)
        
        with open(self.training_data_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            row = features + [accessory_name]
            writer.writerow(row)
            
    def suggest_accessories(self, features):
        """Predicts and suggests accessories based on facial features."""
        if not self.is_trained:
            return []
        feature_vector = [
            features.get('face_shape', 0), features.get('skin_tone', 1), features.get('eye_size', 1),
            features.get('face_ratio', 1.0), features.get('symmetry', 1.0)
        ]
        if not self.feature_db: return []
        feature_vector.extend([0] * (len(self.feature_db[0]) - len(feature_vector)))
        
        feature_vector = self.scaler.transform([feature_vector])
        distances, indices = self.knn_model.kneighbors(feature_vector)
        suggestions = []
        for idx in indices[0]:
            if idx < len(self.accessory_db):
                suggestions.append(self.accessory_db[idx])
        return list(dict.fromkeys(suggestions))

    def get_facial_landmarks(self, frame, face):
        """Gets facial landmarks using the dlib predictor."""
        if self.landmark_available:
            try:
                landmarks = self.predictor(frame, face)
                return [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
            except:
                return self.estimate_landmarks(face.left(), face.top(), face.width(), face.height())
        else:
            return self.estimate_landmarks(face.left(), face.top(), face.width(), h = face.height())

    def estimate_landmarks(self, x, y, w, h):
        """Estimates facial landmarks as a fallback if dlib predictor is unavailable."""
        landmarks = []
        for i in range(68):
            landmarks.append((x + int(w * (i % 8) / 8), y + int(h * (i // 8) / 8)))
        return landmarks

    def extract_facial_features(self, frame, face):
        """Extracts key facial features as a vector."""
        landmarks = self.get_facial_landmarks(frame, face)
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        features = {}
        features['face_shape'] = 0; features['skin_tone'] = 1; features['eye_size'] = 1
        features['face_ratio'] = w / h if h > 0 else 1.0; features['symmetry'] = 1.0
        return features, [features['face_shape'], features['skin_tone'], features['eye_size'], features['face_ratio'], features['symmetry']]

class TrainingModeWindow:
    def __init__(self, master, app):
        self.master = master
        self.app = app
        self.window = tk.Toplevel(self.master)
        self.window.title("FaceFit - Training Mode")
        self.window.geometry("1000x600")
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.source_mode = 'camera'
        self.current_frame = None
        self.cap = None
        self.face_features_vector = None
        
        self.create_ui()
        self.set_source_mode('camera')

    def create_ui(self):
        left_frame = ttk.Frame(self.window)
        left_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(left_frame, width=640, height=480, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        right_frame = ttk.Frame(self.window)
        right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

        ttk.Label(right_frame, text="Input Source:", font=("Helvetica", 12)).pack(pady=(0, 5))
        btn_live = ttk.Button(right_frame, text="Live Camera", command=lambda: self.set_source_mode('camera'))
        btn_live.pack(pady=5, fill=tk.X)
        btn_photo = ttk.Button(right_frame, text="Upload Photo", command=lambda: self.set_source_mode('photo'))
        btn_photo.pack(pady=5, fill=tk.X)
        
        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=10)
        
        ttk.Label(right_frame, text="Select Correct Accessory:", font=("Helvetica", 12)).pack(pady=(0, 5))
        self.accessory_listbox = tk.Listbox(right_frame, height=8)
        for acc in self.app.accessory_images.keys():
            self.accessory_listbox.insert(tk.END, acc)
        self.accessory_listbox.pack(pady=5, fill=tk.X)
        
        btn_save = ttk.Button(right_frame, text="Save Data Point", command=self.save_data_point)
        btn_save.pack(pady=10, fill=tk.X)

    def set_source_mode(self, source_mode):
        if self.source_mode == 'camera' and self.cap:
            self.cap.release()
            self.cap = None
        
        self.source_mode = source_mode
        self.current_frame = None

        if source_mode == 'camera':
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Could not open camera.")
                self.on_closing()
                return
            self.update_frame()
        elif source_mode == 'photo':
            file_path = filedialog.askopenfilename(
                title="Select a photo",
                filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
            )
            if file_path:
                self.current_frame = cv2.imread(file_path)
                self.update_frame()
            else:
                self.set_source_mode('camera')

    def save_data_point(self):
        selected_index = self.accessory_listbox.curselection()
        if not selected_index:
            messagebox.showwarning("No Selection", "Please select an accessory to save.")
            return
        
        if self.face_features_vector is None:
            messagebox.showwarning("No Face Detected", "Please make sure a face is detected before saving.")
            return
            
        accessory_name = self.accessory_listbox.get(selected_index[0])
        self.app.save_data_point(self.face_features_vector, accessory_name)
        messagebox.showinfo("Success", f"Data point saved: {accessory_name}")

    def update_frame(self):
        if self.source_mode == 'camera':
            ret, frame = self.cap.read()
            if not ret:
                self.on_closing()
                return
            self.current_frame = frame
        
        if self.current_frame is None:
            self.window.after(10, self.update_frame)
            return

        frame = self.current_frame.copy()
        frame = cv2.flip(frame, 1)
        faces = self.app.detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        display_frame = frame.copy()

        if faces:
            features, self.face_features_vector = self.app.extract_facial_features(frame, faces[0])
            display_frame = self.visualize_analysis(display_frame, faces[0], features)
        else:
            self.face_features_vector = None

        self.show_frame(display_frame)

        if self.source_mode == 'camera':
            self.window.after(10, self.update_frame)
    
    def show_frame(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=img_tk, anchor=tk.NW)
        self.canvas.image = img_tk
        
    def on_closing(self):
        if self.cap:
            self.cap.release()
        self.window.destroy()
        
    def visualize_analysis(self, frame, face, features):
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        landmarks = self.app.get_facial_landmarks(frame, face)
        for point in landmarks:
            cv2.circle(frame, point, 2, (0, 0, 255), -1)
        
        face_shapes = ["Oval", "Round", "Square", "Heart"]
        cv2.putText(frame, f"Shape: {face_shapes[features['face_shape']]}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame
        
class ModeWindow:
    def __init__(self, master, app, mode):
        self.master = master
        self.app = app
        self.mode = mode
        self.window = tk.Toplevel(self.master)
        self.window.title(f"FaceFit - {self.mode.capitalize()} Mode")
        self.window.geometry("1000x600")
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.source_mode = 'camera'
        self.current_frame = None
        self.cap = None
        self.accessory_to_try_on = None
        
        self.create_ui()
        self.set_source_mode('camera')

    def create_ui(self):
        left_frame = ttk.Frame(self.window)
        left_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(left_frame, width=640, height=480, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        right_frame = ttk.Frame(self.window)
        right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

        ttk.Label(right_frame, text="Input Source:", font=("Helvetica", 12)).pack(pady=(0, 5))
        btn_live = ttk.Button(right_frame, text="Live Camera", command=lambda: self.set_source_mode('camera'))
        btn_live.pack(pady=5, fill=tk.X)
        btn_photo = ttk.Button(right_frame, text="Upload Photo", command=lambda: self.set_source_mode('photo'))
        btn_photo.pack(pady=5, fill=tk.X)

        if self.mode == 'suggestion':
            ttk.Label(right_frame, text="Suggestions:", font=("Helvetica", 12)).pack(pady=(15, 5))
            self.suggestion_listbox = tk.Listbox(right_frame, height=8)
            self.suggestion_listbox.pack(pady=5, fill=tk.X)
            self.suggestion_listbox.bind("<<ListboxSelect>>", self.on_suggestion_select)
        elif self.mode == 'try-on':
            ttk.Label(right_frame, text="Try-On Options:", font=("Helvetica", 12)).pack(pady=(15, 5))
            self.try_on_listbox = tk.Listbox(right_frame, height=8)
            for acc in self.app.accessory_images.keys():
                self.try_on_listbox.insert(tk.END, acc)
            self.try_on_listbox.pack(pady=5, fill=tk.X)
            self.try_on_listbox.bind("<<ListboxSelect>>", self.on_try_on_select)
    
    def set_source_mode(self, source_mode):
        if self.source_mode == 'camera' and self.cap:
            self.cap.release()
            self.cap = None
        
        self.source_mode = source_mode
        self.current_frame = None

        if source_mode == 'camera':
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Could not open camera.")
                self.on_closing()
                return
            self.update_frame()
        elif source_mode == 'photo':
            file_path = filedialog.askopenfilename(
                title="Select a photo",
                filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
            )
            if file_path:
                self.current_frame = cv2.imread(file_path)
                self.update_frame()
            else:
                self.set_source_mode('camera')

    def on_suggestion_select(self, event):
        selected_index = self.suggestion_listbox.curselection()
        if selected_index:
            self.accessory_to_try_on = self.suggestion_listbox.get(selected_index[0])
            self.update_frame()

    def on_try_on_select(self, event):
        selected_index = self.try_on_listbox.curselection()
        if selected_index:
            self.accessory_to_try_on = self.try_on_listbox.get(selected_index[0])
            self.update_frame()

    def update_frame(self):
        if self.source_mode == 'camera':
            ret, frame = self.cap.read()
            if not ret:
                self.on_closing()
                return
            self.current_frame = frame
        
        if self.current_frame is None:
            self.window.after(10, self.update_frame)
            return

        frame = self.current_frame.copy()
        frame = cv2.flip(frame, 1)
        faces = self.app.detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        display_frame = frame.copy()

        if self.mode == 'analysis':
            if faces:
                features, _ = self.app.extract_facial_features(frame, faces[0])
                display_frame = self.visualize_analysis(display_frame, faces[0], features)
        
        elif self.mode == 'suggestion':
            if faces:
                features, _ = self.app.extract_facial_features(frame, faces[0])
                suggestions = self.app.suggest_accessories(features)
                self.update_suggestion_list(suggestions)
                display_frame = self.visualize_suggestions(display_frame, faces[0], suggestions)
                if len(suggestions) > 0 and self.accessory_to_try_on is None:
                    self.accessory_to_try_on = suggestions[0]

        elif self.mode == 'try-on':
            if faces and self.accessory_to_try_on:
                display_frame = self.render_try_on(display_frame, faces[0], self.accessory_to_try_on)
            elif faces:
                cv2.putText(display_frame, "Select an accessory from the list", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
        self.show_frame(display_frame)

        if self.source_mode == 'camera':
            self.window.after(10, self.update_frame)

    def update_suggestion_list(self, suggestions):
        self.suggestion_listbox.delete(0, tk.END)
        for item in suggestions:
            self.suggestion_listbox.insert(tk.END, item)
    
    def show_frame(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=img_tk, anchor=tk.NW)
        self.canvas.image = img_tk
        
    def on_closing(self):
        if self.cap:
            self.cap.release()
        self.window.destroy()
        
    def visualize_analysis(self, frame, face, features):
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        landmarks = self.app.get_facial_landmarks(frame, face)
        for point in landmarks:
            cv2.circle(frame, point, 2, (0, 0, 255), -1)
        
        face_shapes = ["Oval", "Round", "Square", "Heart"]
        cv2.putText(frame, f"Shape: {face_shapes[features['face_shape']]}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame
        
    def visualize_suggestions(self, frame, face, suggestions):
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return frame

    def render_try_on(self, frame, face, accessory_name):
        """Renders a real accessory image onto the face."""
        
        landmarks = self.app.get_facial_landmarks(frame, face)
        if not landmarks:
            return frame

        accessory_info = self.app.accessory_images.get(accessory_name)
        if not accessory_info or not os.path.exists(accessory_info['path']):
            cv2.putText(frame, f"Image for {accessory_name} not found!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return frame

        accessory_type = accessory_info['type']
        accessory_img = cv2.imread(accessory_info['path'], cv2.IMREAD_UNCHANGED)
        
        if accessory_type == 'glasses':
            # Calculate glasses position and size based on eye landmarks
            left_eye_center = tuple(np.mean([landmarks[36], landmarks[37], landmarks[38], landmarks[39]], axis=0).astype(int))
            right_eye_center = tuple(np.mean([landmarks[42], landmarks[43], landmarks[44], landmarks[45]], axis=0).astype(int))
            eye_distance = np.linalg.norm(np.array(left_eye_center) - np.array(right_eye_center))
            
            # Resize glasses to fit face
            new_width = int(eye_distance * 2.5)
            ratio = new_width / accessory_img.shape[1]
            new_height = int(accessory_img.shape[0] * ratio)
            
            accessory_img = cv2.resize(accessory_img, (new_width, new_height))
            
            # Calculate position to overlay
            center_x = (left_eye_center[0] + right_eye_center[0]) // 2
            center_y = (left_eye_center[1] + right_eye_center[1]) // 2
            
            # Overlay the image
            x1, y1 = center_x - new_width // 2, center_y - new_height // 2
            x2, y2 = x1 + new_width, y1 + new_height
            
            if accessory_img.shape[2] == 4: # Has alpha channel
                frame = overlay_transparent_image(frame, accessory_img, (x1, y1))
            else:
                frame[y1:y2, x1:x2] = accessory_img

        elif accessory_type == 'earrings':
            # Place earrings on the ears
            left_ear_pos = landmarks[1]
            right_ear_pos = landmarks[15]
            
            earring_size = int(face.width() * 0.1)
            accessory_img = cv2.resize(accessory_img, (earring_size, earring_size))
            
            # Overlay left earring
            x1_l, y1_l = left_ear_pos[0] - earring_size // 2, left_ear_pos[1]
            frame = overlay_transparent_image(frame, accessory_img, (x1_l, y1_l))
            
            # Overlay right earring
            x1_r, y1_r = right_ear_pos[0] - earring_size // 2, right_ear_pos[1]
            frame = overlay_transparent_image(frame, accessory_img, (x1_r, y1_r))

        elif accessory_type == 'necklace':
            # Place necklace on the neck
            neck_pos = landmarks[8]
            
            neck_width = int(face.width() * 0.8)
            ratio = neck_width / accessory_img.shape[1]
            neck_height = int(accessory_img.shape[0] * ratio)

            accessory_img = cv2.resize(accessory_img, (neck_width, neck_height))

            x1, y1 = neck_pos[0] - neck_width // 2, neck_pos[1] + 20
            frame = overlay_transparent_image(frame, accessory_img, (x1, y1))

        return frame

def overlay_transparent_image(background, overlay, position):
    x, y = position
    h, w, _ = overlay.shape

    # Ensure overlay is within background boundaries
    if x + w > background.shape[1] or y + h > background.shape[0]:
        return background

    # Extract alpha channel
    alpha_s = overlay[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    # Overlay with alpha blending
    for c in range(0, 3):
        background[y:y+h, x:x+w, c] = (alpha_s * overlay[:, :, c] + alpha_l * background[y:y+h, x:x+w, c])

    return background

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = AIAccessorySuggestorApp(root)
        root.mainloop()
    except Exception as e:
        print(f"An error occurred: {e}")