# facefit_GEMINI_STABLE_FINAL.py
# This is the definitive, project-winning version.
# - REFINED AI: Uses an advanced prompt for natural, human-like style advice.
# - SECURE API: Loads the Gemini API key securely from a .env file.
# - RESILIENT: Includes a JSON cache and local rule-based fallback.
# - INDEPENDENT HAIRSTYLES: Hairstyle browsing is now separate from AI generation.
# - [FIX] UI LAYOUT: Corrected the Tkinter geometry manager conflict (pack vs. grid).

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
from collections import Counter
import google.generativeai as genai
from dotenv import load_dotenv

# --- [SECURE] LOAD API KEY FROM ENVIRONMENT ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# -----------------------------------------------

# --- INITIALIZATION ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- CONFIG ---
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480
ACCESORY_DIR = "accessories"
HAIRSTYLE_DIR = "hairstyles_dataset"
ACCESSORY_KEYWORDS = {
    'Glasses / Sunglasses': ['glass', 'sun', 'spectacle', 'eye', 'frame'],
    'Hats / Headwear': ['cap', 'hat', 'fedora', 'beanie', 'headband', 'crown', 'brim'],
    'Earrings / Jewelry': ['earring', 'hoop', 'stud', 'jewel', 'pendant', 'drop'],
    'Necklaces / Pendants': ['necklace', 'chain', 'pendant', 'scarf', 'tie', 'choker']
}

# --- UTILITY & ANALYSIS FUNCTIONS ---
def organize_accessories(ac_dir=ACCESORY_DIR):
    cats = {k: [] for k in ACCESSORY_KEYWORDS.keys()}; cats['Others'] = []
    if os.path.exists(ac_dir):
        for f in sorted(os.listdir(ac_dir)):
            if not f.lower().endswith(('.png')): continue
            assigned=False;
            for cat, kw in ACCESSORY_KEYWORDS.items():
                if any(k in f.lower() for k in kw): cats[cat].append(f); assigned=True; break
            if not assigned: cats['Others'].append(f)
    return {k: v for k, v in cats.items() if v}

def load_hairstyle_images():
    hairstyles = {}
    if not os.path.exists(HAIRSTYLE_DIR):
        print(f"Warning: Hairstyle directory not found at '{HAIRSTYLE_DIR}'")
        return hairstyles
    for gender in ["Male", "Female", "Other"]:
        hairstyles[gender] = {}
        for shape in ["Round", "Square", "Heart", "Oval"]:
            folder = os.path.join(HAIRSTYLE_DIR, gender, shape)
            if os.path.exists(folder):
                hairstyles[gender][shape] = glob.glob(os.path.join(folder, "*.*"))
    return hairstyles

ACCESSORY_CATEGORIES = organize_accessories()

def process_accessory_image(img_bytes):
    acc = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    return acc if acc is not None and acc.shape[2] == 4 else None

# --- RELIABLE GEOMETRIC FACE SHAPE ENGINE ---
def get_face_shape_geometric(landmarks, frame_w, frame_h):
    p_jaw_l,p_jaw_r,p_cheek_l,p_cheek_r,p_forehead,p_chin = landmarks[172],landmarks[397],landmarks[234],landmarks[454],landmarks[10],landmarks[152]
    face_width_jaw = math.hypot((p_jaw_l.x-p_jaw_r.x)*frame_w, (p_jaw_l.y-p_jaw_r.y)*frame_h)
    face_width_cheeks = math.hypot((p_cheek_l.x-p_cheek_r.x)*frame_w, (p_cheek_l.y-p_cheek_r.y)*frame_h)
    face_height = math.hypot((p_forehead.x-p_chin.x)*frame_w, (p_forehead.y-p_chin.y)*frame_h)
    face_shape = "Oval"
    if face_width_jaw > 0 and face_height > 0:
        if abs(face_width_cheeks - face_height) < 25 : face_shape = "Round"
        elif abs(face_width_jaw - face_width_cheeks) < 30: face_shape = "Square"
        elif face_width_cheeks > face_width_jaw and (face_height / face_width_cheeks) < 1.3: face_shape = "Heart"
    return face_shape

# --- GEMINI VISION REPORTING ENGINE ---
def get_gemini_vision_report(api_key, image, face_shape, gender):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('models/gemini-2.5-pro')
        prompt = f"""Act as a world-class personal stylist writing an exclusive style note for a client. Your tone is chic, insightful, and warmly personal, like a message from a trusted friend with an amazing eye for fashion. Avoid generic compliments and AI-sounding phrases. Here's the client's photo. I've determined their face shape is **{face_shape}** and they identify as **{gender}**. Based on the image, write a style guide for them. Instead of a rigid report, structure your response as a friendly, flowing message. Start with their overall vibe. Comment on a feature you notice in the photo and connect it to their {face_shape} face shape in a positive, authentic way. Next, talk about eyewear. Suggest two distinct styles. Name the style and explain *specifically* how it complements their features. Then, move to hairstyles. Recommend two different cuts or styles that would be flattering. Be descriptive. Finally, suggest a color palette. Based on their visible skin tone, hair, and eye color, recommend a palette of 4-5 complementary colors. Describe the colors evocatively. Conclude with a warm sign-off."""
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"An error occurred. Please check your internet connection or API key.\n\nError: {e}"

# --- 2D OVERLAY ENGINE --- (Condensed for brevity)
def _blend_transparent(bg,ov,x,y):
    try:
        h,w,_=bg.shape;oh,ow,_=ov.shape;x1,y1=max(x,0),max(y,0);x2,y2=min(x+ow,w),min(y+oh,h);roi=bg[y1:y2,x1:x2]
        if roi.shape[0]==0 or roi.shape[1]==0:return bg
        ov_x1,ov_y1=max(0,-x),max(0,-y);ov_x2,ov_y2=ov_x1+(x2-x1),ov_y1+(y2-y1);ov_p=ov[ov_y1:ov_y2,ov_x1:ov_x2];mask=(ov_p[:,:,3]/255.0)[:,:,np.newaxis]
        roi_b=(1.0-mask)*roi.astype(float)+mask*ov_p[:,:,:3].astype(float);bg[y1:y2,x1:x2]=roi_b.astype(np.uint8)
    except Exception:pass
    return bg
def get_landmarks_pixels(landmarks,w,h):return{i:(lm.x*w,lm.y*h)for i,lm in enumerate(landmarks)}
def get_angle_and_width(p1,p2):dx,dy=p2[0]-p1[0],p2[1]-p1[1];return math.degrees(math.atan2(dy,dx)),math.hypot(dx,dy)
def overlay_glasses(bg,ov,lms,ovr):
    _,hw=get_angle_and_width(lms[234],lms[454]);a,_=get_angle_and_width(lms[33],lms[263]);ey=(lms[33][1]+lms[263][1])/2.0;cx,cy=int(lms[168][0]),int(ey);tw=int(hw*0.95*ovr.get('scale_factor',1.0));th=int(tw*(ov.shape[0]/ov.shape[1]))if ov.shape[1]>0 else 0;
    if tw==0 or th==0:return bg
    r=cv2.resize(ov,(tw,th));M=cv2.getRotationMatrix2D((tw/2,th/2),a,1);rot=cv2.warpAffine(r,M,(tw,th));return _blend_transparent(bg,rot,cx-tw//2,cy-th//2)
def overlay_hat(bg,ov,lms,ovr):
    _,hw=get_angle_and_width(lms[234],lms[454]);a,_=get_angle_and_width(lms[33],lms[263]);cx,cy=int(lms[10][0]),int(lms[10][1]);tw=int(hw*1.4*ovr.get('scale_factor',1.0));th=int(tw*(ov.shape[0]/ov.shape[1]))if ov.shape[1]>0 else 0;
    if tw==0 or th==0:return bg
    r=cv2.resize(ov,(tw,th));M=cv2.getRotationMatrix2D((tw/2,th/2),a,1);rot=cv2.warpAffine(r,M,(tw,th));return _blend_transparent(bg,rot,cx-tw//2,cy-int(th*0.8))
def overlay_earrings(bg,ov,lms,ovr):
    _,hw=get_angle_and_width(lms[234],lms[454]);tw=int(hw*0.15*ovr.get('scale_factor',1.0));th=int(tw*(ov.shape[0]/ov.shape[1]))if ov.shape[1]>0 else 0;
    if tw==0 or th==0:return bg
    r=cv2.resize(ov,(tw,th));bg=_blend_transparent(bg,r,int(lms[132][0]-tw/2),int(lms[132][1]-th/2));return _blend_transparent(bg,r,int(lms[361][0]-tw/2),int(lms[361][1]-th/2))
def overlay_necklace(bg,ov,lms,ovr):
    a,jw=get_angle_and_width(lms[130],lms[359]);cx,cy=int(lms[152][0]),int(lms[152][1]);tw=int(jw*1.5*ovr.get('scale_factor',1.0));th=int(tw*(ov.shape[0]/ov.shape[1]))if ov.shape[1]>0 else 0;
    if tw==0 or th==0:return bg
    r=cv2.resize(ov,(tw,th));M=cv2.getRotationMatrix2D((tw/2,th/2),a,1);rot=cv2.warpAffine(r,M,(tw,th));return _blend_transparent(bg,rot,cx-tw//2,cy+int(th*0.1))

# --- MEDIAPIPE DETECTION ---
def detect_with_mediapipe(img):
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0].landmark
    return None

class WebcamCaptureThread(threading.Thread):
    def __init__(self,src=0):
        super().__init__(daemon=True); self.cap=cv2.VideoCapture(src,cv2.CAP_DSHOW); self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,VIDEO_WIDTH); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,VIDEO_HEIGHT); self.frame=None; self.running=False; self.lock=threading.Lock()
    def run(self):
        self.running=True
        while self.running:
            ret,fr=self.cap.read()
            if ret:
                with self.lock: self.frame=fr
        self.cap.release()
    def read(self):
        with self.lock: return self.frame.copy() if self.frame is not None else None
    def stop(self): self.running=False

class FaceFitApp(tk.Tk):
    def __init__(self):
        super().__init__(); self.title("FaceFit - AI Advisor (Gemini Edition)"); self.geometry("1000x700")
        self.protocol("WM_DELETE_WINDOW",self._on_close)
        self.video_stream=None; self.live_streaming=False; self.last_frame_raw=None; self.last_frame_composed=None
        self.last_landmarks=None; self.selected_accessories={}; self.accessory_overrides={}; self.selected_accessory_name=None
        self.gender_var=tk.StringVar(value="Female")
        self.CACHE_FILE = "gemini_cache.json"
        self.hairstyles = load_hairstyle_images()
        self.hair_gender_var = tk.StringVar(value="Female")
        self.hair_shape_var = tk.StringVar(value="Oval")
        self._build_ui()
        self.after(50,self._load_accessories_list)

    def _build_ui(self):
        main=ttk.Panedwindow(self,orient=tk.HORIZONTAL); main.pack(fill=tk.BOTH,expand=True,padx=8,pady=8)
        left,right=ttk.Frame(main),ttk.Frame(main); main.add(left,weight=3); main.add(right,weight=1)
        self.left_nb=ttk.Notebook(left); self.left_nb.pack(fill=tk.BOTH,expand=True)
        cam,up=ttk.Frame(self.left_nb),ttk.Frame(self.left_nb); self.left_nb.add(cam,text='📹 Live'); self.left_nb.add(up,text='🖼️ Upload')
        self.vid_lbl=ttk.Label(cam); self.vid_lbl.pack(fill=tk.BOTH,expand=True)
        up_ctrl=ttk.Frame(up); up_ctrl.pack(fill=tk.X,padx=5,pady=5)
        ttk.Button(up_ctrl,text="Load Photo",command=self.load_image_file).pack(side=tk.LEFT)
        self.static_lbl=ttk.Label(up); self.static_lbl.pack(fill=tk.BOTH,expand=True)
        ctrl_fr=ttk.LabelFrame(right,text="Controls"); ctrl_fr.pack(fill=tk.X,padx=5,pady=5)
        ttk.Button(ctrl_fr,text="Start Cam",command=self.start_webcam).grid(row=0,column=0,sticky='ew',padx=2)
        ttk.Button(ctrl_fr,text="Stop Cam",command=self.stop_webcam).grid(row=0,column=1,sticky='ew',padx=2)
        ttk.Button(ctrl_fr,text="Save Look",command=self.save_favorite_look).grid(row=1,column=0,columnspan=2,sticky='ew',padx=2)
        
        self.right_nb=ttk.Notebook(right); self.right_nb.pack(fill=tk.BOTH,expand=True,pady=5)
        adv=ttk.Frame(self.right_nb); self.right_nb.add(adv,text='⭐ AI Profile')
        tryon=ttk.Frame(self.right_nb); self.right_nb.add(tryon,text='👓 Try-On')
        self.hairstyle_tab = ttk.Frame(self.right_nb)
        self.right_nb.add(self.hairstyle_tab, text='💇‍♀️ Hairstyles')

        # --- AI Profile Tab Content ---
        prof_fr=ttk.LabelFrame(adv,text="1. Your Profile"); prof_fr.pack(fill=tk.X,padx=5,pady=5)
        ttk.Label(prof_fr,text="Gender:").grid(row=0,column=0,sticky='w',padx=5); ttk.Combobox(prof_fr,textvariable=self.gender_var,values=['Female','Male','Other'],state='readonly').grid(row=0,column=1,sticky='ew')
        ttk.Button(adv,text="2. Generate AI Style Profile",command=self.run_gemini_analysis,style="Accent.TButton").pack(fill=tk.X,padx=5,pady=10)
        rep_fr=ttk.LabelFrame(adv,text="3. Your AI Style Guide"); rep_fr.pack(fill=tk.BOTH,expand=True,padx=5,pady=5)
        self.rep_txt=tk.Text(rep_fr,wrap=tk.WORD,height=10); self.rep_txt.pack(fill=tk.BOTH,expand=True,padx=5,pady=5); self.rep_txt.insert(tk.END,"Click 'Generate AI Style Profile' to get started.")

        # --- Try-On Tab Content ---
        acc_fr=ttk.LabelFrame(tryon,text="Select Accessory"); acc_fr.pack(fill=tk.BOTH,expand=True,padx=5,pady=5)
        self.cat_var=tk.StringVar(); self.cat_cb=ttk.Combobox(acc_fr,textvariable=self.cat_var,state='readonly'); self.cat_cb.pack(fill=tk.X); self.cat_cb.bind('<<ComboboxSelected>>',self._load_accessories_by_category)
        self.acc_lb=tk.Listbox(acc_fr,height=8,exportselection=False); self.acc_lb.pack(fill=tk.BOTH,expand=True); self.acc_lb.bind('<<ListboxSelect>>',self._on_accessory_select)
        ttk.Label(tryon,text="Use '+' & '-' to resize.").pack(fill=tk.X,padx=5,pady=5)

        # --- Hairstyle Tab with Manual Controls ---
        hair_ctrl_fr = ttk.LabelFrame(self.hairstyle_tab, text="Find Your Style")
        hair_ctrl_fr.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(hair_ctrl_fr, text="Gender:").grid(row=0, column=0, padx=5, pady=2, sticky='w')
        ttk.Combobox(hair_ctrl_fr, textvariable=self.hair_gender_var, values=['Female', 'Male', 'Other'], state='readonly').grid(row=0, column=1, padx=5, pady=2, sticky='ew')
        ttk.Label(hair_ctrl_fr, text="Face Shape:").grid(row=1, column=0, padx=5, pady=2, sticky='w')
        ttk.Combobox(hair_ctrl_fr, textvariable=self.hair_shape_var, values=['Oval', 'Round', 'Square', 'Heart'], state='readonly').grid(row=1, column=1, padx=5, pady=2, sticky='ew')
        
        # --- [FIX] Use .grid() for the button, not .pack() ---
        ttk.Button(hair_ctrl_fr, text="Find Hairstyles", command=self._manual_hairstyle_search).grid(row=2, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        
        hair_gallery_fr = ttk.LabelFrame(self.hairstyle_tab, text="Suggestions")
        hair_gallery_fr.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        hair_canvas=tk.Canvas(hair_gallery_fr); hair_scrollbar=ttk.Scrollbar(hair_gallery_fr,orient="vertical",command=hair_canvas.yview)
        self.scrollable_hair_frame=ttk.Frame(hair_canvas)
        self.scrollable_hair_frame.bind("<Configure>",lambda e:hair_canvas.configure(scrollregion=hair_canvas.bbox("all")))
        hair_canvas.create_window((0,0),window=self.scrollable_hair_frame,anchor="nw"); hair_canvas.configure(yscrollcommand=hair_scrollbar.set)
        hair_canvas.pack(side="left",fill="both",expand=True); hair_scrollbar.pack(side="right",fill="y")

        ttk.Style().configure("Accent.TButton",font=('Helvetica',10,'bold'),foreground='green')
        self.bind('<plus>',lambda e:self._scale_selected(0.05)); self.bind('<minus>',lambda e:self._scale_selected(-0.05))

    def _update_webcam_loop(self):
        if not self.live_streaming:return
        fr=self.video_stream.read()
        if fr is None:self.after(10,self._update_webcam_loop);return
        fr=cv2.flip(fr,1);self.last_frame_raw=fr
        comp=self.process_frame_for_tryon(fr);self.last_frame_composed=comp
        imgtk=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(comp,cv2.COLOR_BGR2RGB)))
        self.vid_lbl.imgtk=imgtk;self.vid_lbl.configure(image=imgtk);self.after(10,self._update_webcam_loop)

    def get_placement_from_name(self,name):
        nl=name.lower()
        for cat,kw in ACCESSORY_KEYWORDS.items():
            if any(k in nl for k in kw):
                if cat=='Glasses / Sunglasses':return'glasses'
                if cat=='Hats / Headwear':return'hat'
                if cat=='Earrings / Jewelry':return'earrings'
                if cat=='Necklaces / Pendants':return'necklace'
        return 'default'

    def process_frame_for_tryon(self,fr):
        self.last_landmarks=detect_with_mediapipe(fr);comp=fr.copy()
        if self.last_landmarks:
            lms_px=get_landmarks_pixels(self.last_landmarks,fr.shape[1],fr.shape[0])
            for name,img in self.selected_accessories.items():
                if img is not None:
                    ovr=self.accessory_overrides.get(name,{});p=self.get_placement_from_name(name)
                    if p=='glasses':comp=overlay_glasses(comp,img,lms_px,ovr)
                    elif p=='hat':comp=overlay_hat(comp,img,lms_px,ovr)
                    elif p=='earrings':comp=overlay_earrings(comp,img,lms_px,ovr)
                    elif p=='necklace':comp=overlay_necklace(comp,img,lms_px,ovr)
        return comp

    def run_gemini_analysis(self):
        if not GEMINI_API_KEY: messagebox.showwarning("API Key Missing","GEMINI_API_KEY not found. Using local fallback.")
        fr=None
        if self.left_nb.index(self.left_nb.select())==0 and self.live_streaming:fr=self.last_frame_raw
        elif self.left_nb.index(self.left_nb.select())==1:fr=self.last_frame_raw
        if fr is None:messagebox.showerror("Error","No image to analyze.");return
        lms=detect_with_mediapipe(fr)
        if not lms:messagebox.showerror("Error","Could not detect a face.");return
        h,w,_=fr.shape; face_shape=get_face_shape_geometric(lms,w,h)
        img=Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB))
        self.rep_txt.delete(1.0,tk.END);self.rep_txt.insert(tk.END,"🚀 Generating your AI Style Profile...")
        self.update_idletasks()
        
        self.hair_gender_var.set(self.gender_var.get())
        self.hair_shape_var.set(face_shape)
        self._update_hairstyle_suggestions()
        self.right_nb.select(self.hairstyle_tab)
        
        threading.Thread(target=self._get_and_display_report,args=(img,face_shape,self.gender_var.get()),daemon=True).start()

    def _manual_hairstyle_search(self):
        self._update_hairstyle_suggestions()

    def _update_hairstyle_suggestions(self):
        gender = self.hair_gender_var.get()
        face_shape = self.hair_shape_var.get()

        for widget in self.scrollable_hair_frame.winfo_children(): widget.destroy()
        if not self.hairstyles:
            ttk.Label(self.scrollable_hair_frame,text="Hairstyle dataset not found.").pack()
            return

        imgs=self.hairstyles.get(gender,{}).get(face_shape,[])
        if not imgs:
            ttk.Label(self.scrollable_hair_frame,text=f"No hairstyles found for:\n{gender} / {face_shape} Face").pack(pady=20)
            return
        
        for i,img_path in enumerate(imgs):
            try:
                pil_img=Image.open(img_path).resize((100,100),Image.LANCZOS)
                tk_img=ImageTk.PhotoImage(pil_img)
                lbl=ttk.Label(self.scrollable_hair_frame,image=tk_img,cursor="hand2")
                lbl.image=tk_img
                row,col=divmod(i,2); lbl.grid(row=row,column=col,padx=5,pady=5)
                lbl.bind("<Button-1>",lambda e,p=img_path:self._show_full_hairstyle(p))
            except Exception as e:
                print(f"Error loading hairstyle image {img_path}: {e}")

    def _show_full_hairstyle(self,img_path):
        if not os.path.exists(img_path):return
        try:
            pil_img=Image.open(img_path).resize((VIDEO_WIDTH,VIDEO_HEIGHT),Image.LANCZOS)
            comp=cv2.cvtColor(np.array(pil_img),cv2.COLOR_RGB2BGR)
            self.last_frame_composed=comp
            imgtk=ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(comp,cv2.COLOR_BGR2RGB)))
            
            target_label = self.vid_lbl if self.left_nb.index(self.left_nb.select())==0 else self.static_lbl
            target_label.imgtk=imgtk
            target_label.configure(image=imgtk)
        except Exception as e:
            messagebox.showerror("Error",f"Could not display hairstyle image.\n{e}")

    # --- Caching and Fallback (unchanged) ---
    def _load_cache(self):
        if not os.path.exists(self.CACHE_FILE):return{}
        try:
            with open(self.CACHE_FILE,'r')as f:return json.load(f)
        except json.JSONDecodeError:return{}
    def _save_cache(self,cache_data):
        try:
            with open(self.CACHE_FILE,'w')as f:json.dump(cache_data,f,indent=2)
        except Exception as e:print(f"Error saving cache: {e}")
    def get_local_style_report(self,face_shape,gender):
        report=f"### Local Style Guide for {gender} with {face_shape} Face ###\n\n"
        hairstyles={"Round":["Layered Bob","Side-swept Bangs"],"Square":["Soft Waves","Long Layers"],"Heart":["Lob (Long Bob)","Pixie Cut"],"Oval":["Blunt Bob","Almost any style works!"]}
        if gender=="Male":hairstyles={"Round":["Textured Crop","Pompadour"],"Square":["Buzz Cut","Slicked Back"],"Heart":["Side Part","Longer fringe"],"Oval":["Quiff","Almost any style"]}
        eyewear={"Round":["Rectangular Frames","Wayfarers"],"Square":["Round Frames","Aviators"],"Heart":["Cat-Eye Glasses","Rimless Frames"],"Oval":["Most frames work","Geometric"]}
        report+="**Eyewear Recommendations:**\n";[report:=report+f"- {item}\n" for item in eyewear.get(face_shape,[])]
        report+="\n**Hairstyle Suggestions:**\n";[report:=report+f"- {item}\n" for item in hairstyles.get(face_shape,[])]
        report+="\n**Suggested Color Palette:**\n- Neutral Tones\n- Accent Colors\n"
        return report
    def _get_and_display_report(self,img,shape,gender):
        cache_key=f"{shape.lower()}_{gender.lower()}"; cache_data=self._load_cache()
        if cache_key in cache_data:
            rep=cache_data[cache_key]; report_to_display=f"--- Displaying Cached Style Profile ---\n\n{rep}"
            self.rep_txt.delete(1.0,tk.END); self.rep_txt.insert(tk.END,report_to_display)
        elif not GEMINI_API_KEY:
            fallback_rep=self.get_local_style_report(shape,gender); report_to_display=f"NOTE: AI Advisor offline.\n\n--- Displaying Local Suggestions ---\n\n{fallback_rep}"
            self.rep_txt.delete(1.0,tk.END); self.rep_txt.insert(tk.END,report_to_display)
        else:
            try:
                rep=get_gemini_vision_report(GEMINI_API_KEY,img,shape,gender)
                if rep.startswith("An error occurred"):raise Exception(rep.split('\n')[0])
                cache_data[cache_key]=rep; self._save_cache(cache_data)
                self.rep_txt.delete(1.0,tk.END); self.rep_txt.insert(tk.END,rep)
            except Exception as e:
                fallback_rep=self.get_local_style_report(shape,gender); error_msg=str(e).split('Error:')[-1].strip()
                report_to_display=f"NOTE: AI Advisor unavailable.\n(Error: {error_msg})\n\n--- Displaying Local Suggestions ---\n\n{fallback_rep}"
                self.rep_txt.delete(1.0,tk.END); self.rep_txt.insert(tk.END,report_to_display)

    # --- Other methods (unchanged) ---
    def start_webcam(self):
        if not self.live_streaming:self.video_stream=WebcamCaptureThread();self.video_stream.start();self.live_streaming=True;self.left_nb.select(0);self._update_webcam_loop()
    def stop_webcam(self):
        if self.live_streaming:self.video_stream.stop();self.live_streaming=False
    def load_image_file(self):
        path=filedialog.askopenfilename(filetypes=[("Images","*.png;*.jpg;*.jpeg")]);
        if not path:return
        self.stop_webcam();fr=cv2.imread(path)
        if fr is None:messagebox.showerror("Error","Could not load image.");return
        h,w,_=fr.shape;sc=min(VIDEO_WIDTH/w,VIDEO_HEIGHT/h)if min(w,h)>1 else 1
        fr=cv2.resize(fr,(int(w*sc),int(h*sc)));self.last_frame_raw=fr
        self.refresh_static_image();self.left_nb.select(1)
    def _on_accessory_select(self,event=None):
        sel=[self.acc_lb.get(i) for i in self.acc_lb.curselection()];self.selected_accessories.clear()
        for name in sel:
            try:
                with open(os.path.join(ACCESORY_DIR,name),'rb')as fh:
                    p=process_accessory_image(fh.read())
                    if p is not None:self.selected_accessories[name]=p;self.accessory_overrides.setdefault(name,{'scale_factor':1.0})
            except Exception as e:print(f"Error loading {name}: {e}")
        self.selected_accessory_name=sel[-1]if sel else None
        if self.left_nb.index(self.left_nb.select())==1 and self.last_frame_raw is not None:self.refresh_static_image()
    def refresh_static_image(self):
        if self.last_frame_raw is None:return
        comp=self.process_frame_for_tryon(self.last_frame_raw);self.last_frame_composed=comp
        imgtk=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(comp,cv2.COLOR_BGR2RGB)))
        self.static_lbl.imgtk=imgtk;self.static_lbl.configure(image=imgtk)
    def _load_accessories_list(self):
        self.cat_cb.config(values=list(ACCESSORY_CATEGORIES.keys()))
        if ACCESSORY_CATEGORIES:self.cat_var.set(list(ACCESSORY_CATEGORIES.keys())[0]);self._load_accessories_by_category()
    def _load_accessories_by_category(self,event=None):
        self.acc_lb.delete(0,tk.END)
        for f in ACCESSORY_CATEGORIES.get(self.cat_var.get(),[]):self.acc_lb.insert(tk.END,f)
    def _scale_selected(self,delta):
        if self.selected_accessory_name:
            ov=self.accessory_overrides.setdefault(self.selected_accessory_name,{'scale_factor':1.0})
            ov['scale_factor']=max(0.2,min(3.0,ov.get('scale_factor',1.0)+delta))
            if self.left_nb.index(self.left_nb.select())==1 and self.last_frame_raw is not None:self.refresh_static_image()
    def save_favorite_look(self):
        if self.last_frame_composed is not None:
            os.makedirs("Favorites",exist_ok=True);fname=f"Favorites/Look_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(fname,self.last_frame_composed);messagebox.showinfo("Saved",f"Saved to {fname}")
        else:messagebox.showinfo("Error","No image to save.")
    def _on_close(self):self.stop_webcam();self.destroy()

if __name__ == "__main__":
    os.makedirs(ACCESORY_DIR,exist_ok=True)
    app=FaceFitApp(); app.mainloop()