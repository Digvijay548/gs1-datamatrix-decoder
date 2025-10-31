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
model = YOLO(r"modifiedwith1d.pt").to(device)
_ = model(torch.zeros(1, 3, 416, 416))
torch.set_grad_enabled(False)

# ==== GUI setup ====
root = tk.Tk()
root.title("Fast DataMatrix Detection & Decoding")
root.geometry("1000x850")

panel = None
image_path = None
decoded_texts = tk.Text(root, height=12, width=100)
decoded_texts.pack(pady=10)

def browse_image():
    """Browse and show selected image."""
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

def detect_and_decode_threaded():
    """Run detection in background to keep GUI responsive."""
    thread = threading.Thread(target=detect_and_decode)
    thread.start()

def decode_datamatrix(i, img, x1, y1, x2, y2):
    """Helper function for decoding each cropped image."""
    cropped = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    decoded = pylibdmtx.decode(gray)
    if decoded:
        decoded_str = decoded[0].data.decode("utf-8", errors="ignore")
        return i, decoded_str, True
    return i, None, False

def detect_and_decode():
    """Run YOLO detection + DataMatrix decoding."""
    global image_path, panel
    if not image_path:
        decoded_texts.insert(tk.END, "⚠️ Please select an image first!\n")
        return
    
    decoded_texts.delete(1.0, tk.END)
    decoded_texts.insert(tk.END, "🔍 Running detection...\n")
    img = cv2.imread(image_path)

    # YOLO inference
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    start_time = time.perf_counter()
    results = model(img_rgb, imgsz=640, verbose=False)   # ←
    result = results[0]
    end_time2 = time.perf_counter()
    total_time2 = end_time2 - start_time
    decoded_texts.insert(tk.END, f"\n⏱️ Total detection time: {total_time2:.6f} seconds\n")
    
    if img is None:
        decoded_texts.insert(tk.END, "❌ Could not read image!\n")
        return

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        decoded_texts.insert(tk.END, "⚠️ No objects detected.\n")
        return

    decoded_count = 0
    

    decode_tasks = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i, box in enumerate(boxes):
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue

            expand = 5
            x1 = max(0, int(x1) - expand)
            y1 = max(0, int(y1) - expand)
            x2 = min(img.shape[1] - 1, int(x2) + expand)
            y2 = min(img.shape[0] - 1, int(y2) + expand)

            decode_tasks.append(executor.submit(decode_datamatrix, i, img, x1, y1, x2, y2))

        for future in concurrent.futures.as_completed(decode_tasks):
            i, decoded_str, success = future.result()
            xyxy = boxes[i].xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            expand = 15
            x1 = max(0, int(x1) - expand)
            y1 = max(0, int(y1) - expand)
            x2 = min(img.shape[1] - 1, int(x2) + expand)
            y2 = min(img.shape[0] - 1, int(y2) + expand)

            if success:
                decoded_count += 1
                decoded_texts.insert(tk.END, f"✅ Object {i+1}: {decoded_str}\n")
                print(f"✅ Decoded {i+1}: {decoded_str}")
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
    if decoded_count == 0:
        decoded_texts.insert(tk.END, "⚠️ No DataMatrix codes decoded.\n")

    # Update GUI
    annotated = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    annotated = cv2.resize(annotated, (800, 500))
    im_pil = Image.fromarray(annotated)
    tk_img = ImageTk.PhotoImage(im_pil)
    panel.configure(image=tk_img)
    panel.image = tk_img

    decoded_texts.insert(tk.END, f"\n✅ Done. Total decoded: {decoded_count}\n")

# ==== Buttons ====
frame = tk.Frame(root)
frame.pack(pady=10)

tk.Button(frame, text="Browse Image", command=browse_image, width=20, height=2, bg="#3498db", fg="white").grid(row=0, column=0, padx=20)
tk.Button(frame, text="Detect & Decode", command=detect_and_decode_threaded, width=20, height=2, bg="#27ae60", fg="white").grid(row=0, column=1, padx=20)

root.mainloop()
