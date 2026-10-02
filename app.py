import cv2
import numpy as np
import gradio as gr
import time
import os
import glob
import torch

from src.roi_detector import IndustrialROIDetector
from src.anomaly_detector import PatchCoreAnomalyDetector

# Initialize core modules
roi_detector = IndustrialROIDetector()
anomaly_detector = PatchCoreAnomalyDetector()

# Fit memory bank on standard MVTec good dataset
def init_memory():
    train_dir = "data/bottle/train/good"
    if os.path.exists(train_dir):
        image_paths = glob.glob(os.path.join(train_dir, "*.png"))
        good_crops = []
        for p in image_paths[:10]: # Standard reference subset
            img = cv2.imread(p)
            if img is not None:
                crop, _, _ = roi_detector.extract_roi(img, save_crop=False)
                good_crops.append(crop if crop is not None else img)
        if good_crops:
            anomaly_detector.fit_memory_bank(good_crops)
            print("✅ Memory bank baseline loaded successfully.")

try:
    init_memory()
except Exception as e:
    print(f"⚠️ Memory setup warning: {e}")

def inspect_image(input_image, conf_thresh, anomaly_thresh):
    if input_image is None:
        return None, None, None, "Please upload an image."
    
    start_time = time.time()
    
    # 1. ROI Detection
    roi_detector.conf_thresh = conf_thresh
    roi_crop, bbox, is_fallback = roi_detector.extract_roi(input_image, save_crop=False)
    
    # Handle boolean conversion for array returns
    if isinstance(is_fallback, (np.ndarray, torch.Tensor)):
        is_fallback = bool(is_fallback.any())
    else:
        is_fallback = bool(is_fallback)

    if roi_crop is None or roi_crop.size == 0:
        roi_crop = input_image.copy()
        bbox = (0, 0, input_image.shape[1], input_image.shape[0])
        is_fallback = True

    # 2. Frame 1: Bounding Box Detection Overlay
    annotated_frame = input_image.copy()
    if bbox:
        x, y, w, h = bbox
        color = (255, 165, 0) if is_fallback else (0, 255, 0)
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 3)
        label = "ROI (Fallback)" if is_fallback else "ROI (YOLO)"
        cv2.putText(annotated_frame, label, (x, max(y - 10, 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # 3. Anomaly Prediction Engine
    anomaly_score, raw_heatmap, _ = anomaly_detector.predict_anomaly(roi_crop)
    
    # Frame 2: Raw Heatmap (Grayscale / Jet map)
    if len(raw_heatmap.shape) == 2 or raw_heatmap.shape[2] == 1:
        raw_heatmap_norm = cv2.normalize(raw_heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        colored_raw = cv2.applyColorMap(raw_heatmap_norm, cv2.COLORMAP_JET)
    else:
        colored_raw = raw_heatmap

    # Frame 3: Overlay Heatmap (Blended with ROI Crop)
    roi_resized = cv2.resize(roi_crop, (colored_raw.shape[1], colored_raw.shape[0]))
    overlay_heatmap = cv2.addWeighted(roi_resized, 0.5, colored_raw, 0.5, 0)
    
    # Inspection Report
    is_defective = anomaly_score > anomaly_thresh
    status_str = "❌ DEFECT DETECTED" if is_defective else "✅ NORMAL (PASS)"
    latency_ms = (time.time() - start_time) * 1000

    report = f"""
    ### 📋 Inspection Status: {status_str}
    - **Anomaly Score**: `{anomaly_score:.4f}`
    - **Threshold**: `{anomaly_thresh:.2f}`
    - **Processing Time**: `{latency_ms:.2f} ms`
    - **ROI Mode**: `{"Contour Fallback" if is_fallback else "YOLOv8 Dynamic"}`
    """
    
    return annotated_frame, colored_raw, overlay_heatmap, report

# 3-Frame Layout UI
with gr.Blocks(title="Industrial Defect Inspection System") as demo:
    gr.Markdown("# 🏭 Industrial Surface Defect Inspection Dashboard")
    gr.Markdown("Two-stage real-time inspection pipeline using **YOLOv8 Dynamic Cropping** and **PatchCore Feature Memory Engine**.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="numpy", label="Upload Assembly Line Image")
            conf_slider = gr.Slider(minimum=0.1, maximum=0.9, value=0.25, step=0.05, label="YOLO Confidence Threshold")
            thresh_slider = gr.Slider(minimum=0.5, maximum=15.0, value=3.5, step=0.1, label="Anomaly Score Threshold")
            btn = gr.Button("🔍 Run Inspection", variant="primary")
            
        with gr.Column(scale=2):
            with gr.Row():
                output_bbox = gr.Image(label="1. Bounding Box ROI Detection")
                output_raw_map = gr.Image(label="2. Raw Feature Anomaly Map")
            with gr.Row():
                output_overlay_map = gr.Image(label="3. Heatmap Defect Localization Overlay")
            
            output_report = gr.Markdown()
            
    btn.click(
        fn=inspect_image,
        inputs=[input_img, conf_slider, thresh_slider],
        outputs=[output_bbox, output_raw_map, output_overlay_map, output_report]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
