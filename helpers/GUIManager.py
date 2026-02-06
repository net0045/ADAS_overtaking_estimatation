import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

class GUIManager(ctk.CTk):
    def __init__(self, window_name="ADAS System", width=1280, height=800):
        super().__init__()
        
        self.title(window_name)
        self.geometry(f"{width}x{height}")
        self.protocol("WM_DELETE_WINDOW", self.close)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1) # Video
        self.grid_rowconfigure(1, weight=0) # Control Panel

        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Control Panel
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.grid(row=1, column=0, padx=10, pady=20, sticky="ew")
        
        self.speed_label = ctk.CTkLabel(self.control_frame, text="Ego Speed: 90 km/h", font=("Arial", 14, "bold"))
        self.speed_label.pack(pady=(10, 0))

        self.speed_slider = ctk.CTkSlider(
            self.control_frame, 
            from_=0, 
            to=180, 
            number_of_steps=180,
            command=self._update_speed_label
        )
        self.speed_slider.set(90)
        self.speed_slider.pack(padx=20, pady=10, fill="x")

        self.current_speed_ms = 90 / 3.6
        self.running = True

    def _update_speed_label(self, value):
        self.speed_label.configure(text=f"Ego Speed: {int(value)} km/h")
        self.current_speed_ms = value / 3.6

    def display_window(self, frame):
        if not self.running:
            return

        window_width = self.video_label.winfo_width()
        window_height = self.video_label.winfo_height()

        # Original image dimensions
        img_h, img_w = frame.shape[:2]
        
        if window_width > 10 and window_height > 10:
            aspect_ratio = img_w / img_h
            
            new_w = window_width
            new_h = int(new_w / aspect_ratio)
            
            if new_h > window_height:
                new_h = window_height
                new_w = int(new_h * aspect_ratio)
            
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        self.video_label.configure(image=img_tk)
        self.video_label.image = img_tk 
        
        self.update_idletasks()
        self.update()

    def get_ego_speed(self):
        return self.current_speed_ms

    def close(self):
        self.running = False
        self.destroy()
        cv2.destroyAllWindows()