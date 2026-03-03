import tkinter as tk
from tkinter import ttk, messagebox, filedialog  # Added filedialog
import cv2
from PIL import Image, ImageTk

class SudokuApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sudoku Solver")
        self.root.geometry("500x300")
        # UI
        self.label = ttk.Label(root, text="Upload a Sudoku puzzle", font=("Arial", 12))
        self.label.pack(pady=20)
        
        self.upload_btn = ttk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(pady=10)

        self.path_label = ttk.Label(root, text="No file selected", wraplength=400)
        self.path_label.pack(pady=10)

    def upload_image(self):
        # Open a file dialog to select an image
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.path_label.config(text=f"Selected: {file_path}")
            self.process_puzzle(file_path)

    def process_puzzle(self, image_path):
        print(f"Loading image: {image_path}")
        
        # Load image
        img = cv2.imread(image_path)
        # Basic Preprocessing 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        messagebox.showinfo("Success", "Image uploaded")

if __name__ == "__main__":
    root = tk.Tk()
    app = SudokuApp(root)
    root.mainloop()