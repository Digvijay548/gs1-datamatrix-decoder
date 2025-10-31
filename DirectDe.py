from ultralytics import YOLO
import torch

# 1️⃣ Load your trained YOLO model
model = YOLO(r"D:\Final_python_poc\modifiedwith1d.pt")

# 2️⃣ Export it as TorchScript (for CPU use)
model.export(format="torchscript", imgsz=640, dynamic=False, device='cpu')

print("✅ Export complete! TorchScript file created successfully.")
