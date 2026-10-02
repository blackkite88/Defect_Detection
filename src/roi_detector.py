import cv2
import numpy as np

class IndustrialROIDetector:
    def __init__(self, conf_thresh=0.25):
        self.conf_thresh = conf_thresh
        try:
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')
        except Exception:
            self.model = None

    def extract_roi(self, image, save_crop=False):
        if image is None:
            return None, None, True

        h, w = image.shape[:2]
        
        # 1. Primary Component Segmenter: Extract full physical object body
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image.copy()
        
        # Otsu thresholding separates the bottle object from dark assembly background
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > (h * w * 0.05):  # Ensure it's the main object
                x, y, bw, bh = cv2.boundingRect(c)
                
                # Add padding around the component
                pad = 10
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
                
                roi = image[y1:y2, x1:x2]
                bbox = (x1, y1, x2 - x1, y2 - y1)
                
                # If YOLO is used, check if it detected the full frame or object
                is_fallback = False
                return roi, bbox, is_fallback

        # 2. Fallback to Full Frame if no valid contour
        return image.copy(), (0, 0, w, h), True
