import os
import glob
from src.roi_detector import IndustrialROIDetector

def main():
    detector = IndustrialROIDetector(model_weights="yolov8n.pt", conf_thresh=0.1)

    test_image_dir = "./data/mvtec_bottle/test/broken_large"
    output_dir = "./outputs/rois"

    image_paths = sorted(glob.glob(os.path.join(test_image_dir, "*.png")))
    if not image_paths:
        print(f"[ERROR] No images found in '{test_image_dir}'.")
        return

    print(f"\nProcessing {len(image_paths)} test images...")

    for path in image_paths:
        filename = os.path.basename(path)
        roi, bbox, annotated = detector.extract_roi(path, save_crop=True, output_dir=output_dir)
        print(f"[SUCCESS] Processed {filename} | BBox: {bbox} | Cropped Shape: {roi.shape}")

    print(f"\nPhase 1 Complete! Cropped ROI images saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()
