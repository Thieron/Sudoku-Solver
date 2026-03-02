import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk

class SudokuApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sudoku Solver")
        self.root.geometry("400x200")

        # Layout
        self.label = ttk.Label(root, text="Upload a Sudoku Pizzle", font=("Arial", 12))
        self.label.pack(pady=20)
        self.run_btn = ttk.Button(root, text="Upload", command=self.run_logic)
        self.run_btn.pack(pady=10)

    def run_logic(self):
        print("running app")
        # will work on 

if __name__ == "__main__":
    root = tk.Tk()
    app = SudokuApp(root)
    print("Application is running. Check your taskbar for the window!")
    root.mainloop()