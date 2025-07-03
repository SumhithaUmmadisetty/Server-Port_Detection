import streamlit as st
import os
import cv2
from ultralytics import YOLO

# === Setup Paths ===
BASE_MODEL = "yolov8n.pt"
RETRAIN_MODEL = "Server Ports Detection/runs/detect/train/weights/last.pt"
DATA_YAML = "Server Ports Detection/data.yaml"
IMG_DIR = "Server Ports Detection/data/images"
TRAIN_IMG_DIR = f"{IMG_DIR}/train"
UPLOAD_DIR = f"{IMG_DIR}/uploaded"
LABEL_DIR = "Server Ports Detection/data/labels/train"

# === Load YOLO model ===
model_path = RETRAIN_MODEL if os.path.exists(RETRAIN_MODEL) else BASE_MODEL
model = YOLO(model_path)

st.title("🔌 YOLOv8 Server Port Self-Learning App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Inference
    image = cv2.imread(image_path)
    results = model(image)[0]

    # Save pseudo-label to label dir (in YOLO format)
    label_file = os.path.join(LABEL_DIR, uploaded_file.name.replace(".jpg", ".txt"))
    with open(label_file, "w") as f:
        for box in results.boxes:
            cls = int(box.cls.item())
            x, y, w, h = box.xywhn[0]
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    # Move image to training folder
    train_img_path = os.path.join(TRAIN_IMG_DIR, uploaded_file.name)
    os.rename(image_path, train_img_path)

    # Display detections
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    image = cv2.resize(image, None, fx=2, fy=2)
    st.image(image, channels="BGR", caption="Detected Ports")

    # Retrain
    if st.button("🔁 Retrain Model"):
        with st.spinner("Training..."):
            model.train(data=DATA_YAML, epochs=3, batch=1, imgsz=640)
            st.success("✅ Retraining complete!")
