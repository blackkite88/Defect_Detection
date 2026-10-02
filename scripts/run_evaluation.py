import os
import cv2
import numpy as np
import torch
from src.roi_detector import IndustrialROIDetector
from src.anomaly_detector import PatchCoreAnomalyDetector

def evaluate_mvtec_category(category_path, conf_thresh=0.25, anomaly_thresh=3.5):
    print(f"\n==========================================")
    print(f" 🧪 Evaluating Category: {os.path.basename(category_path).upper()}")
    print(f"==========================================")

    roi_detector = IndustrialROIDetector(conf_thresh=conf_thresh)
    anomaly_detector = PatchCoreAnomalyDetector()

    # 1. Fit Memory Bank on Training Images (Normal Samples)
    train_dir = os.path.join(category_path, "train", "good")
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".png")]
    
    print(f"[Eval] Extracting baseline memory bank from {len(train_files)} clean training images...")
    ref_crops = []
    for p in train_files:
        img = cv2.imread(p)
        crop, _, _ = roi_detector.extract_roi(img, save_crop=False)
        ref_crops.append(crop)
    
    anomaly_detector.fit_memory_bank(ref_crops)

    # 2. Evaluate Test Set (Good vs. Defective)
    test_dir = os.path.join(category_path, "test")
    y_true = [] # 0 for normal, 1 for defect
    y_scores = []

    print("[Eval] Running inference on test set...")
    for subfolder in os.listdir(test_dir):
        sub_path = os.path.join(test_dir, subfolder)
        if not os.path.isdir(sub_path):
            continue
        
        is_defect_folder = 0 if subfolder == "good" else 1
        img_files = [os.path.join(sub_path, f) for f in os.listdir(sub_path) if f.endswith(".png")]

        for img_p in img_files:
            img = cv2.imread(img_p)
            crop, _, _ = roi_detector.extract_roi(img, save_crop=False)
            score, _, _ = anomaly_detector.predict_anomaly(crop)

            y_true.append(is_defect_folder)
            y_scores.append(score)

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Calculate basic classification metrics
    preds = (y_scores > anomaly_thresh).astype(int)
    accuracy = (preds == y_true).mean() * 100
    
    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))
    tn = np.sum((preds == 0) & (y_true == 0))

    print(f"\n📊 --- Evaluation Results ---")
    print(f"- Total Test Samples : {len(y_true)}")
    print(f"- Accuracy           : {accuracy:.2f}%")
    print(f"- True Positives (TP): {tp}")
    print(f"- False Positives(FP): {fp}")
    print(f"- True Negatives (TN): {tn}")
    print(f"- False Negatives(FN): {fn}")
    print(f"- Avg Normal Score   : {np.mean(y_scores[y_true == 0]):.4f}")
    print(f"- Avg Defect Score   : {np.mean(y_scores[y_true == 1]):.4f}")

if __name__ == "__main__":
    if os.path.exists("data/bottle"):
        evaluate_mvtec_category("data/bottle")
