import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class HandDrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Drawing App")
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        
        # Setup video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Drawing variables
        self.color_points = {
            'blue': [deque(maxlen=1024)],
            'green': [deque(maxlen=1024)],
            'red': [deque(maxlen=1024)],
            'yellow': [deque(maxlen=1024)],
            'black': [deque(maxlen=1024)]
        }
        self.color_indices = {color: 0 for color in self.color_points}
        self.colors = {
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'black': (0, 0, 0)
        }
        self.current_color = 'black'
        self.brush_size = 5
        
        # Create blank canvas
        self.canvas_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        self.drawing_mode = True
        
        # Setup UI
        self.setup_ui()
        
        # Start video loop
        self.update()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel (webcam feed)
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.webcam_label = ttk.Label(left_panel)
        self.webcam_label.pack()
        
        # Right panel (controls and canvas)
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Control frame
        control_frame = ttk.LabelFrame(right_panel, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=5)
        
        # Color buttons
        color_frame = ttk.Frame(control_frame)
        color_frame.pack(fill=tk.X, pady=5)
        
        for color in ['black', 'blue', 'green', 'red', 'yellow']:
            btn = tk.Button(
                color_frame, 
                text=color.capitalize(), 
                bg=color if color != 'black' else 'black',
                fg='white' if color == 'black' else 'black',
                command=lambda c=color: self.set_color(c),
                width=8
            )
            btn.pack(side=tk.LEFT, padx=5)
        
        # Brush size
        size_frame = ttk.Frame(control_frame)
        size_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(size_frame, text="Brush Size:").pack(side=tk.LEFT)
        self.size_slider = ttk.Scale(size_frame, from_=1, to=20, value=5, command=self.set_brush_size)
        self.size_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Mode buttons
        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(fill=tk.X, pady=5)
        
        self.draw_btn = tk.Button(
            mode_frame, 
            text="Drawing Mode", 
            bg='lightgreen',
            command=self.set_drawing_mode
        )
        self.draw_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = tk.Button(
            mode_frame, 
            text="Clear Canvas", 
            bg='lightcoral',
            command=self.clear_canvas
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Canvas frame
        canvas_frame = ttk.LabelFrame(right_panel, text="Drawing Canvas", padding="10")
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas_label = ttk.Label(canvas_frame)
        self.canvas_label.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    def set_color(self, color):
        self.current_color = color
        self.status_var.set(f"Selected color: {color}")
    
    def set_brush_size(self, val):
        self.brush_size = int(float(val))
    
    def set_drawing_mode(self):
        self.drawing_mode = not self.drawing_mode
        if self.drawing_mode:
            self.draw_btn.config(text="Drawing Mode", bg='lightgreen')
            self.status_var.set("Drawing mode: ON")
        else:
            self.draw_btn.config(text="Navigation Mode", bg='lightgray')
            self.status_var.set("Drawing mode: OFF")
    
    def clear_canvas(self):
        self.canvas_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        for color in self.color_points:
            self.color_points[color] = [deque(maxlen=1024)]
            self.color_indices[color] = 0
        self.status_var.set("Canvas cleared")
    
    def update(self):
        ret, frame = self.cap.read()
        if ret:
            # Flip and convert color
            frame = cv2.flip(frame, 1)
            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand landmarks
            result = self.hands.process(framergb)
            
            if result.multi_hand_landmarks:
                landmarks = []
                for handslms in result.multi_hand_landmarks:
                    for lm in handslms.landmark:
                        lmx = int(lm.x * frame.shape[1])
                        lmy = int(lm.y * frame.shape[0])
                        landmarks.append([lmx, lmy])
                    
                    self.mp_draw.draw_landmarks(frame, handslms, self.mp_hands.HAND_CONNECTIONS)
                
                fore_finger = (landmarks[8][0], landmarks[8][1])
                thumb = (landmarks[4][0], landmarks[4][1])
                cv2.circle(frame, fore_finger, 3, (0, 255, 0), -1)
                
                # Calculate distance between thumb and index finger
                distance = ((thumb[0] - fore_finger[0])**2 + (thumb[1] - fore_finger[1])**2)**0.5
                
                if distance < 30:  # Pinch gesture
                    if self.drawing_mode:
                        for color in self.color_points:
                            self.color_points[color].append(deque(maxlen=1024))
                            self.color_indices[color] += 1
                
                elif fore_finger[1] <= 65:  # Top area (for potential UI)
                    pass  # We have Tkinter buttons now
                
                elif self.drawing_mode:
                    self.color_points[self.current_color][self.color_indices[self.current_color]].appendleft(fore_finger)
            
            # Draw all lines on frame and canvas
            for color, points in self.color_points.items():
                for stroke in points:
                    for i in range(1, len(stroke)):
                        if stroke[i-1] is None or stroke[i] is None:
                            continue
                        cv2.line(frame, stroke[i-1], stroke[i], self.colors[color], self.brush_size)
                        cv2.line(self.canvas_img, stroke[i-1], stroke[i], self.colors[color], self.brush_size)
            
            # Update webcam feed
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.webcam_label.imgtk = imgtk
            self.webcam_label.configure(image=imgtk)
            
            # Update canvas
            canvas_display = cv2.cvtColor(self.canvas_img, cv2.COLOR_BGR2RGB)
            canvas_img = Image.fromarray(canvas_display)
            canvas_imgtk = ImageTk.PhotoImage(image=canvas_img)
            self.canvas_label.imgtk = canvas_imgtk
            self.canvas_label.configure(image=canvas_imgtk)
        
        self.root.after(10, self.update)
    
    def on_closing(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = HandDrawingApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
