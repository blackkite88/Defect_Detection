import os
import glob
import cv2
import numpy as np
from src.anomaly_detector import PatchCoreAnomalyDetector

def main():
    roi_dir = "./outputs/rois"
    output_dir = "./outputs/anomaly_maps"
    os.makedirs(output_dir, exist_ok=True)

    roi_paths = sorted(glob.glob(os.path.join(roi_dir, "*.png")))
    if not roi_paths:
        print(f"[ERROR] No ROI images found in '{roi_dir}'. Make sure Phase 1 ran properly.")
        return

    print(f"Loaded {len(roi_paths)} cropped ROIs for Anomaly Detection...")

    # Initialize Phase 2 Anomaly Engine
    detector = PatchCoreAnomalyDetector(backbone_name="resnet18")

    # Use first few images as normal background baseline
    ref_images = [cv2.imread(p) for p in roi_paths[:2]]
    detector.fit_memory_bank(ref_images)

    print("\n--- Running Anomaly Segmentation on ROIs ---")
    for path in roi_paths:
        filename = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            continue

        score, norm_map, heatmap = detector.predict_anomaly(img)

        # Blend original ROI with colorized heatmap overlay
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        # Save side-by-side output (Original ROI | Heatmap Overlay)
        combined = np.hstack((img, overlay))
        out_path = os.path.join(output_dir, f"anomaly_{filename}")
        cv2.imwrite(out_path, combined)

        print(f"[SUCCESS] {filename} | Max Anomaly Score: {score:.4f} | Saved: {out_path}")

    print(f"\nPhase 2 Complete! Heatmap overlay images saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()
