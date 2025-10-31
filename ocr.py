import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import pytesseract
import numpy as np
from difflib import SequenceMatcher

# ✅ Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def detect_text_with_tesseract(image):
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    results = []
    for i in range(len(data["text"])):
        text = data["text"][i]
        conf_val = data["conf"][i]
        try:
            conf = float(conf_val)
        except (ValueError, TypeError):
            conf = -1
        if text.strip() != "" and conf > 0:
            results.append({"text": text, "conf": conf})
    return results


class OCRApp:
    def __init__(self, master):
        self.master = master
        master.title("Advanced OCR Desktop App")
        master.geometry("900x600")  # half size window
        master.configure(bg="#1e1e1e")

        self.image_path = None
        self.cv_img = None
        self.display_img = None
        self.tk_img = None
        self.scale_x = self.scale_y = 1
        self.start_x = self.start_y = None
        self.current_rect = None
        self.rois = []
        self.roi_rects = []
        self.passing_score = tk.DoubleVar(value=80)

        self.canvas = tk.Canvas(master, bg="gray", height=400)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

        control_frame = tk.Frame(master, bg="#222")
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        tk.Button(control_frame, text="Load Image", command=self.load_image,
                  bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Run OCR", command=self.run_ocr,
                  bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Clear ROIs", command=self.clear_rois,
                  bg="#f44336", fg="white").pack(side=tk.LEFT, padx=5)

        tk.Label(control_frame, text="Passing Score:", bg="#222", fg="white").pack(side=tk.LEFT, padx=10)
        self.slider = tk.Scale(control_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                               variable=self.passing_score, bg="#333", fg="white", length=150)
        self.slider.pack(side=tk.LEFT)

    # ------------------- Core UI & Drawing -------------------
    def load_image(self):
        self.image_path = filedialog.askopenfilename(
            title="Select an image", filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")]
        )
        if not self.image_path:
            return
        self.cv_img = cv2.imread(self.image_path)
        h, w = self.cv_img.shape[:2]
        max_w, max_h = 850, 450
        scale = min(max_w / w, max_h / h)
        self.display_img = cv2.resize(self.cv_img, (int(w * scale), int(h * scale)))
        self.scale_x = w / (w * scale)
        self.scale_y = h / (h * scale)
        self.display_image(self.display_img)

    def display_image(self, cv_img):
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv_img_rgb)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        self.canvas.image = self.tk_img

    def on_canvas_click(self, event):
      # Detect click on existing ROI (decode only clicked one)
      for i, roi in enumerate(self.rois):
          x1, y1, x2, y2 = roi
          if x1 <= event.x <= x2 and y1 <= event.y <= y2:
              self.decode_roi(i)
              return
  
      # Otherwise start new ROI selection
      self.start_x, self.start_y = event.x, event.y
      self.current_rect = self.canvas.create_rectangle(
          event.x, event.y, event.x, event.y, outline="yellow", width=2
      )


    def on_canvas_drag(self, event):
        if self.current_rect:
            self.canvas.coords(self.current_rect, self.start_x, self.start_y, event.x, event.y)

    def on_canvas_release(self, event):
        if self.current_rect:
            x1, y1, x2, y2 = self.canvas.coords(self.current_rect)
            roi = (int(x1), int(y1), int(x2), int(y2))
            self.rois.append(roi)
            self.roi_rects.append(self.current_rect)
            self.current_rect = None

    # ------------------- OCR Processing -------------------
    def run_ocr(self):
       if not self.rois:
           messagebox.showinfo("Info", "Please draw ROI(s) first.")
           return
       # ✅ Decode only the last drawn ROI
       last_index = len(self.rois) - 1
       self.decode_roi(last_index)


    def decode_roi(self, i):
        if self.cv_img is None or i >= len(self.rois):
            return
        x1, y1, x2, y2 = self.rois[i]
        real_x1, real_y1 = int(x1 * self.scale_x), int(y1 * self.scale_y)
        real_x2, real_y2 = int(x2 * self.scale_x), int(y2 * self.scale_y)
        roi_img = self.cv_img[real_y1:real_y2, real_x1:real_x2]

        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        results = detect_text_with_tesseract(gray)
        decoded_text = " ".join([r["text"] for r in results if r["text"].strip() != ""])
        self.show_expected_input(decoded_text, i)

    # ------------------- Popup Comparison -------------------
    def show_expected_input(self, decoded_text, roi_index):
        popup = tk.Toplevel(self.master)
        popup.title(f"ROI {roi_index + 1} Result")
        popup.geometry("400x320")
        popup.configure(bg="#1e1e1e")

        tk.Label(popup, text="Decoded Text (Copyable):", bg="#1e1e1e", fg="white",
                 font=("Arial", 10, "bold")).pack(pady=5)
        decoded_entry = tk.Text(popup, height=4, wrap="word", bg="#2b2b2b", fg="white")
        decoded_entry.insert(tk.END, decoded_text)
        decoded_entry.pack(padx=10, pady=5, fill=tk.X)

        # ✅ Copy button
        tk.Button(popup, text="📋 Copy", bg="#555", fg="white",
                  command=lambda: self.copy_text(decoded_entry)).pack(pady=5)

        tk.Label(popup, text="Enter Expected Text:", bg="#1e1e1e", fg="white").pack()
        expected_entry = tk.Entry(popup, width=50, bg="#2b2b2b", fg="white")
        expected_entry.pack(pady=5)

        def compare_text():
            user_expected = expected_entry.get().strip()
            score = self.calculate_similarity(decoded_text, user_expected)
            threshold = self.passing_score.get()
            color = "green" if score >= threshold else "red"
            self.canvas.itemconfig(self.roi_rects[roi_index], outline=color, width=3)
            messagebox.showinfo("Result", f"Match Score: {score:.2f}%")

        tk.Button(popup, text="Compare", command=compare_text,
                  bg="#4CAF50", fg="white").pack(pady=10)

    def calculate_similarity(self, text1, text2):
        if not text2:
            return 0
        ratio = SequenceMatcher(None, text1.strip().lower(), text2.strip().lower()).ratio()
        return ratio * 100

    def clear_rois(self):
        for rect_id in self.roi_rects:
            self.canvas.delete(rect_id)
        self.rois.clear()
        self.roi_rects.clear()

    def copy_text(self, text_widget):
        text = text_widget.get("1.0", tk.END).strip()
        self.master.clipboard_clear()
        self.master.clipboard_append(text)
        messagebox.showinfo("Copied", "Decoded text copied to clipboard!")


if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()
