import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
import os

# -----------------------------
# UI Setup (unchanged)
# -----------------------------
root = tk.Tk()
root.title("FaceFit 2D Premium Pro")
root.geometry("900x700")
root.configure(bg="#1e1e1e")

video_label = tk.Label(root, bg="#1e1e1e")
video_label.pack(pady=10)

accessory_path = None

def select_accessory():
    global accessory_path
    accessory_path = filedialog.askopenfilename(
        title="Select Accessory Image",
        filetypes=[("Image Files", "*.png")]
    )
    if accessory_path:
        messagebox.showinfo("Selected", f"Accessory loaded:\n{os.path.basename(accessory_path)}")

# -----------------------------
# LIVE TRY-ON FUNCTION
# -----------------------------
def start_tryon():
    import math
    import time

    global video_label, accessory_path

    if not accessory_path or not os.path.exists(accessory_path):
        messagebox.showerror("Error", "Please select a valid accessory image first.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to access webcam.")
        return

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

    # ✅ Load accessory only once
    accessory_img = cv2.imread(accessory_path, cv2.IMREAD_UNCHANGED)
    if accessory_img is None:
        messagebox.showerror("Error", f"Failed to load {accessory_path}")
        cap.release()
        return

    # Smoothing buffer for better stability
    center_buffer = []
    smoothing_window = 5

    def overlay_transparent(bg, overlay, x, y, w, h, angle):
        """Overlay accessory with rotation and transparency"""
        if overlay is None or bg is None:
            return bg
        overlay = cv2.resize(overlay, (w, h))
        if overlay.shape[2] == 4:
            # Rotate overlay with alpha
            rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            overlay = cv2.warpAffine(overlay, rot_mat, (w, h))
            mask = overlay[:, :, 3:] / 255.0
            overlay_rgb = overlay[:, :, :3]
            h_, w_, _ = overlay.shape
            if y + h_ > bg.shape[0] or x + w_ > bg.shape[1] or x < 0 or y < 0:
                return bg
            roi = bg[y:y + h_, x:x + w_]
            bg[y:y + h_, x:x + w_] = (roi * (1 - mask) + overlay_rgb * mask).astype(np.uint8)
        return bg

    def update_frame():
        ret, frame = cap.read()
        if not ret:
            video_label.after(10, update_frame)
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            ih, iw, _ = frame.shape
            for face_landmarks in results.multi_face_landmarks:
                pts = np.array([(lm.x * iw, lm.y * ih, lm.z * iw) for lm in face_landmarks.landmark])
                left_eye = pts[33][:2]
                right_eye = pts[263][:2]
                nose = pts[1][:2]

                # Head pose estimation
                face_3d = np.array([
                    pts[33], pts[263], pts[1], pts[61], pts[291], pts[199]
                ])
                face_2d = np.array([
                    pts[33][:2], pts[263][:2], pts[1][:2], pts[61][:2], pts[291][:2], pts[199][:2]
                ], dtype=np.float64)

                focal_length = 1 * iw
                cam_matrix = np.array([[focal_length, 0, iw / 2],
                                       [0, focal_length, ih / 2],
                                       [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                success, rot_vec, trans_vec = cv2.solvePnP(
                    face_3d, face_2d, cam_matrix, dist_matrix, flags=cv2.SOLVEPNP_ITERATIVE
                )
                rmat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                pitch, yaw, roll = [a * 180 for a in angles]

                # Smooth nose center
                center_buffer.append(nose)
                if len(center_buffer) > smoothing_window:
                    center_buffer.pop(0)
                smooth_nose = np.mean(center_buffer, axis=0)

                # Natural scaling and placement
                dist = np.linalg.norm(left_eye - right_eye)
                scale = int(dist * 2.3)
                x = int(smooth_nose[0] - scale / 2)
                y = int(smooth_nose[1] - scale / 1.7)

                frame = overlay_transparent(
                    frame, accessory_img, x, y,
                    scale,
                    int(scale * accessory_img.shape[0] / accessory_img.shape[1]),
                    -roll
                )

        # Convert to Tkinter image
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(img))
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        video_label.after(10, update_frame)

    update_frame()

# -----------------------------
# BUTTONS (unchanged)
# -----------------------------
btn_frame = tk.Frame(root, bg="#1e1e1e")
btn_frame.pack(pady=10)

select_btn = ttk.Button(btn_frame, text="Select Accessory", command=select_accessory)
select_btn.grid(row=0, column=0, padx=10)

start_btn = ttk.Button(btn_frame, text="Start Try-On", command=start_tryon)
start_btn.grid(row=0, column=1, padx=10)

exit_btn = ttk.Button(btn_frame, text="Exit", command=root.destroy)
exit_btn.grid(row=0, column=2, padx=10)

# -----------------------------
# Run App
# -----------------------------
root.mainloop()
