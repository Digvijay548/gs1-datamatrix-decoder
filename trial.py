import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import torch
import threading
from ultralytics import YOLO
from pylibdmtx import pylibdmtx
import time
import concurrent.futures

# ==== Load YOLO model ====
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Using device: {device}")
model = YOLO(r"D:\Final_python_poc\modifiedwith1d.pt").to(device)

# ==== GUI setup ====
root = tk.Tk()
root.title("Fast DataMatrix Detection & Decoding")
root.geometry("1000x900")

panel = None
image_path = None
decoded_texts = tk.Text(root, height=12, width=100)
decoded_texts.pack(pady=10)

# ==== Default global parameters (will be updated by sliders) ====
CONF_THRESHOLD = 0.50
TOLERANCE = 10

# ==== Browse and Load Image ====
def browse_image():
    global image_path, panel
    image_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*"))
    )
    if image_path:
        img = Image.open(image_path).resize((800, 500))
        tk_img = ImageTk.PhotoImage(img)
        if panel is None:
            panel = tk.Label(root, image=tk_img)
            panel.image = tk_img
            panel.pack(pady=10)
        else:
            panel.configure(image=tk_img)
            panel.image = tk_img
        decoded_texts.delete(1.0, tk.END)
        decoded_texts.insert(tk.END, f"📂 Loaded: {image_path}\n")

# ==== Thread wrapper ====
def detect_and_decode_threaded():
    thread = threading.Thread(target=detect_and_decode)
    thread.start()

# ==== Decode helper ====
def decode_datamatrix(i, img, x1, y1, x2, y2):
    cropped = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    decoded = pylibdmtx.decode(gray)
    if decoded:
        decoded_str = decoded[0].data.decode("utf-8", errors="ignore")
        return i, decoded_str, True
    return i, None, False

# ==== Main detection and decoding ====
def detect_and_decode():
    global image_path, panel, CONF_THRESHOLD, TOLERANCE
    if not image_path:
        decoded_texts.insert(tk.END, "⚠️ Please select an image first!\n")
        return
    
    decoded_texts.delete(1.0, tk.END)
    decoded_texts.insert(tk.END, f"🔍 Running detection (threshold={CONF_THRESHOLD:.2f}, tolerance={TOLERANCE}px)...\n")

    img = cv2.imread(image_path)
    if img is None:
        decoded_texts.insert(tk.END, "❌ Could not read image!\n")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    start_time = time.perf_counter()
    results = model(img_rgb, imgsz=640, verbose=False)
    result = results[0]
    end_time2 = time.perf_counter()
    total_time2 = end_time2 - start_time
    decoded_texts.insert(tk.END, f"\n⏱️ Total detection time: {total_time2:.6f} seconds\n")

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        decoded_texts.insert(tk.END, "⚠️ No objects detected.\n")
        return

    decoded_count = 0
    decode_tasks = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i, box in enumerate(boxes):
            conf = float(box.conf[0].cpu().numpy())
            if conf < CONF_THRESHOLD:
                continue

            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            # Apply tolerance expansion
            x1 = max(0, x1 - TOLERANCE)
            y1 = max(0, y1 - TOLERANCE)
            x2 = min(img.shape[1] - 1, x2 + TOLERANCE)
            y2 = min(img.shape[0] - 1, y2 + TOLERANCE)

            decode_tasks.append(executor.submit(decode_datamatrix, i, img, x1, y1, x2, y2))

        for future in concurrent.futures.as_completed(decode_tasks):
            i, decoded_str, success = future.result()
            xyxy = boxes[i].xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            x1 = max(0, x1 - TOLERANCE)
            y1 = max(0, y1 - TOLERANCE)
            x2 = min(img.shape[1] - 1, x2 + TOLERANCE)
            y2 = min(img.shape[0] - 1, y2 + TOLERANCE)

            if success:
                decoded_count += 1
                decoded_texts.insert(tk.END, f"✅ Object {i+1}: {decoded_str}\n")
                color = (0, 255, 0)
                label = decoded_str[:15]
            else:
                decoded_texts.insert(tk.END, f"❌ Object {i+1}: Not decoded\n")
                color = (0, 0, 255)
                label = "Not decoded"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    end_time = time.perf_counter()
    total_time = end_time - start_time
    decoded_texts.insert(tk.END, f"\n⏱️ Total decode time: {total_time:.6f} seconds\n")
    decoded_texts.insert(tk.END, f"\n✅ Done. Total decoded: {decoded_count}\n")

    annotated = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    annotated = cv2.resize(annotated, (800, 500))
    im_pil = Image.fromarray(annotated)
    tk_img = ImageTk.PhotoImage(im_pil)
    panel.configure(image=tk_img)
    panel.image = tk_img

# ==== Button and Slider Frame ====
frame = tk.Frame(root)
frame.pack(pady=10)

tk.Button(frame, text="Browse Image", command=browse_image, width=20, height=2, bg="#3498db", fg="white").grid(row=0, column=0, padx=20)
tk.Button(frame, text="Detect & Decode", command=detect_and_decode_threaded, width=20, height=2, bg="#27ae60", fg="white").grid(row=0, column=1, padx=20)

# ==== Sliders ====
def update_conf(val):
    global CONF_THRESHOLD
    CONF_THRESHOLD = float(val) / 100

def update_tol(val):
    global TOLERANCE
    TOLERANCE = int(val)

slider_frame = tk.Frame(root)
slider_frame.pack(pady=10)

tk.Label(slider_frame, text="Confidence Threshold (0–1):").grid(row=0, column=0, padx=10)
conf_slider = tk.Scale(slider_frame, from_=0, to=100, orient="horizontal", length=300, command=update_conf)
conf_slider.set(50)  # default 0.5
conf_slider.grid(row=0, column=1)

tk.Label(slider_frame, text="Tolerance (px):").grid(row=1, column=0, padx=10)
tol_slider = tk.Scale(slider_frame, from_=0, to=50, orient="horizontal", length=300, command=update_tol)
tol_slider.set(10)  # default 10
tol_slider.grid(row=1, column=1)

root.mainloop()
