import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np

class SudokuApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sudoku Solver")
        self.root.geometry("700x620")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(False, False)

        header = tk.Frame(root, bg="#1e1e2e")
        header.pack(pady=(20, 5))
        tk.Label(header, text="Sudoku Solver",
                 font=("Arial", 22, "bold"), fg="#cdd6f4", bg="#1e1e2e").pack()
        tk.Label(header, text="Upload a photo of a Sudoku puzzle to solve it",
                 font=("Arial", 10), fg="#6c7086", bg="#1e1e2e").pack()

        btn_frame = tk.Frame(root, bg="#1e1e2e")
        btn_frame.pack(pady=12)
        self.upload_btn = tk.Button(
            btn_frame, text="  Upload Image  ", command=self.upload_image,
            font=("Arial", 11, "bold"), bg="#89b4fa", fg="#1e1e2e",
            relief="flat", padx=14, pady=7, cursor="hand2",
            activebackground="#b4befe", activeforeground="#1e1e2e"
        )
        self.upload_btn.pack(side="left", padx=6)

        self.solve_btn = tk.Button(
            btn_frame, text="  Solve  ", command=self.run_solve,
            font=("Arial", 11, "bold"), bg="#a6e3a1", fg="#1e1e2e",
            relief="flat", padx=14, pady=7, cursor="hand2",
            activebackground="#94e2d5", activeforeground="#1e1e2e",
            state="disabled"
        )
        self.solve_btn.pack(side="left", padx=6)

        self.status_label = tk.Label(root, text="No file selected",font=("Arial", 9), fg="#6c7086", bg="#1e1e2e", wraplength=500)
        self.status_label.pack()

        grid_outer = tk.Frame(root, bg="#313244")
        grid_outer.pack(pady=14)

        self.cells = []
        for r in range(9):
            row_cells = []
            for c in range(9):
                pt = 3 if r % 3 == 0 else 1
                pl = 3 if c % 3 == 0 else 1
                pb = 3 if r == 8 else 1
                pr = 3 if c == 8 else 1
                cell_frame = tk.Frame(grid_outer, bg="#313244")
                cell_frame.grid(row=r, column=c, padx=(pl, pr), pady=(pt, pb))
                var = tk.StringVar(value="")
                lbl = tk.Label(cell_frame, textvariable=var,
                               width=3, height=1,
                               font=("Courier", 15, "bold"),
                               bg="#1e1e2e", fg="#cdd6f4", relief="flat")
                lbl.pack()
                row_cells.append((var, lbl))
            self.cells.append(row_cells)

        self._board = None
        self._original_mask = None

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return
        self.status_label.config(text=f"  {file_path}", fg="#6c7086")
        try:
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError("Cannot read image.")
            grid_img = self.extract_grid(img)
            board = self.extract_digits(grid_img)
            self._board = board
            self._original_mask = [[board[r][c] != 0 for c in range(9)] for r in range(9)]
            self._refresh_grid(board)
            self.solve_btn.config(state="normal")
            self.status_label.config(text="Board extracted - press Solve!", fg="#a6e3a1")
        except Exception as e:
            messagebox.showerror("Extraction Error", str(e))
            self.status_label.config(text=f"Error: {e}", fg="#f38ba8")

    def run_solve(self):
        if self._board is None:
            return
        board_copy = [row[:] for row in self._board]
        if self.solve_sudoku(board_copy):
            self._board = board_copy
            self._refresh_grid(board_copy, highlight_solved=True)
            self.status_label.config(text="Solved!", fg="#a6e3a1")
        else:
            messagebox.showerror("No Solution",
                                 "This puzzle has no valid solution.\n"
                                 "Digit extraction may have errors.")
            self.status_label.config(text="No solution found.", fg="#f38ba8")

    def _refresh_grid(self, board, highlight_solved=False):
        for r in range(9):
            for c in range(9):
                var, lbl = self.cells[r][c]
                val = board[r][c]
                var.set(str(val) if val != 0 else "")
                if val == 0:
                    lbl.config(fg="#585b70")
                elif highlight_solved and not self._original_mask[r][c]:
                    lbl.config(fg="#89b4fa")
                else:
                    lbl.config(fg="#cdd6f4")

    def extract_grid(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found - is this a Sudoku image?")
        largest = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
        if len(approx) != 4:
            raise ValueError("Could not find the 4-corner Sudoku grid border.")
        pts = approx.reshape(4, 2)
        rect = self.order_points(pts.astype("float32"))
        side = 450
        dst = np.array([[0,0],[side-1,0],[side-1,side-1],[0,side-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(gray, M, (side, side))
        return warp

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def extract_digits(self, grid_img):
        sudoku = [[0]*9 for _ in range(9)]
        cell_size = grid_img.shape[0] // 9
        _, grid_bin = cv2.threshold(grid_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        for y in range(9):
            for x in range(9):
                cell = grid_bin[y*cell_size:(y+1)*cell_size, x*cell_size:(x+1)*cell_size]
                sudoku[y][x] = self.recognize_digit(cell)
        return sudoku

    def recognize_digit(self, cell_bin):
        h, w = cell_bin.shape
        margin = max(2, int(min(h, w) * 0.12))
        roi = cell_bin[margin:h-margin, margin:w-margin]
        if cv2.countNonZero(roi) < 30:
            return 0
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi)
        if num_labels < 2:
            return 0
        best_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        blob_area = stats[best_label, cv2.CC_STAT_AREA]
        cell_area = roi.shape[0] * roi.shape[1]
        if blob_area < cell_area * 0.04 or blob_area > cell_area * 0.90:
            return 0
        digit_mask = np.uint8(labels == best_label) * 255
        resized = cv2.resize(digit_mask, (28, 28), interpolation=cv2.INTER_AREA)
        return self._classify_digit(resized)

    def _classify_digit(self, img28):
        try:
            return self._knn_classify(img28)
        except Exception:
            return self._heuristic_classify(img28)

    def _build_knn(self):
        if hasattr(self, '_knn_model'):
            return self._knn_model
        samples, labels_list = [], []
        font_faces = [
            cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX,
        ]
        for digit in range(1, 10):
            for font in font_faces:
                for scale in [0.8, 1.0, 1.2]:
                    canvas = np.zeros((28, 28), dtype=np.uint8)
                    text = str(digit)
                    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
                    x = max(0, (28 - tw) // 2)
                    y = max(th, (28 + th) // 2)
                    cv2.putText(canvas, text, (x, y), font, scale, 255, 2, cv2.LINE_AA)
                    samples.append(canvas.flatten().astype(np.float32) / 255.0)
                    labels_list.append(float(digit))
        knn = cv2.ml.KNearest_create()
        knn.train(np.array(samples, dtype=np.float32),
                  cv2.ml.ROW_SAMPLE,
                  np.array(labels_list, dtype=np.float32))
        self._knn_model = knn
        return knn

    def _knn_classify(self, img28):
        knn = self._build_knn()
        sample = img28.flatten().astype(np.float32).reshape(1, -1) / 255.0
        _, results, _, _ = knn.findNearest(sample, k=5)
        return int(results[0][0])

    def _heuristic_classify(self, img28):
        density = cv2.countNonZero(img28) / (28 * 28)
        if density < 0.08:  return 1
        if density > 0.45:  return 8
        if density < 0.15:  return 7
        if density < 0.20:  return 3
        if density < 0.27:  return 2
        if density < 0.35:  return 4
        if density < 0.40:  return 5
        return 6

    def solve_sudoku(self, board):
        empty = self.find_empty(board)
        if not empty:
            return True
        row, col = empty
        for num in range(1, 10):
            if self.is_valid(board, num, row, col):
                board[row][col] = num
                if self.solve_sudoku(board):
                    return True
                board[row][col] = 0
        return False

    def find_empty(self, board):
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    return i, j
        return None

    def is_valid(self, board, num, row, col):
        if num in board[row]:
            return False
        if any(board[i][col] == num for i in range(9)):
            return False
        bx, by = col // 3 * 3, row // 3 * 3
        for i in range(by, by + 3):
            for j in range(bx, bx + 3):
                if board[i][j] == num:
                    return False
        return True


if __name__ == "__main__":
    root = tk.Tk()
    app = SudokuApp(root)
    root.mainloop()