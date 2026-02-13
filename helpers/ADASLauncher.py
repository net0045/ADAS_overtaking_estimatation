import customtkinter as ctk
from tkinter import filedialog

class ADASLauncher(ctk.CTk):
    def __init__(self, start_callback):
        super().__init__()
        self.start_callback = start_callback
        self.title("ADAS System Configuration")
        self.geometry("500x650")
        ctk.set_appearance_mode("dark")

        self.label_file = ctk.CTkLabel(self, text="Video Source", font=("Arial", 16, "bold"))
        self.label_file.pack(pady=(20, 5))
        
        self.file_path = ctk.StringVar(value="No file selected")
        self.btn_browse = ctk.CTkButton(self, text="Browse Video", command=self.browse_file)
        self.btn_browse.pack(pady=5)
        self.label_path = ctk.CTkLabel(self, textvariable=self.file_path, font=("Arial", 10))
        self.label_path.pack()

        self.label_cam = ctk.CTkLabel(self, text="Camera Settings", font=("Arial", 16, "bold"))
        self.label_cam.pack(pady=(30, 5))

        self.slider_fov = self.create_slider("FOV", 40, 180, 140)
        self.slider_horizon = self.create_slider("Horizon Y", 0, 1080, 540)

        self.label_yolo = ctk.CTkLabel(self, text="YOLO Inference Settings", font=("Arial", 16, "bold"))
        self.label_yolo.pack(pady=(30, 5))

        self.label_sz = ctk.CTkLabel(self, text="Inference Resolution (imgsz):")
        self.label_sz.pack()
        self.seg_imgsz = ctk.CTkSegmentedButton(self, values=["320", "640", "1280"])
        self.seg_imgsz.set("640") # Výchozí hodnota
        self.seg_imgsz.pack(pady=10)

        self.slider_thresh = self.create_slider("Confidence Threshold", 0.1, 1.0, 0.45)

        self.btn_start = ctk.CTkButton(
            self, text="LAUNCH ADAS ENGINE", 
            fg_color="#2ecc71", hover_color="#27ae60", 
            font=("Arial", 14, "bold"), height=40,
            command=self.launch
        )
        self.btn_start.pack(pady=40)

    def create_slider(self, text, from_, to, default):
        label = ctk.CTkLabel(self, text=f"{text}: {default}")
        label.pack()
        slider = ctk.CTkSlider(self, from_=from_, to=to, command=lambda v: label.configure(text=f"{text}: {round(v, 2)}"))
        slider.set(default)
        slider.pack(pady=5)
        return slider

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.MOV *.avi")])
        if path:
            self.file_path.set(path)

    def launch(self):
        config = {
            "video_path": self.file_path.get(),
            "fov": float(self.slider_fov.get()),
            "horizon": int(self.slider_horizon.get()), 
            "imgsz": int(self.seg_imgsz.get()),        
            "thresh": float(self.slider_thresh.get())
        }

        if config["video_path"] == "No file selected":
            print("Error: Please select a video file.")
            return
        
        self.withdraw()
        self.quit()
        self.destroy() 
        
        self.start_callback(config)