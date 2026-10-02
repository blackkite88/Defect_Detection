import os
import cv2
import numpy as np

base_dir = "data/bottle"
train_good = os.path.join(base_dir, "train", "good")
test_good = os.path.join(base_dir, "test", "good")
test_defective = os.path.join(base_dir, "test", "broken")

for d in [train_good, test_good, test_defective]:
    os.makedirs(d, exist_ok=True)

def create_base_bottle():
    img = np.full((500, 500, 3), 30, dtype=np.uint8)
    # Body
    cv2.ellipse(img, (250, 300), (80, 180), 0, 0, 360, (180, 180, 180), -1)
    # Cap
    cv2.rectangle(img, (220, 90), (280, 130), (140, 140, 140), -1)
    return img

print("[1/2] Generating clean training & test baseline samples...")
for i in range(10):
    clean_img = create_base_bottle()
    # Add minor noise for variability
    noise = np.random.randint(-5, 5, clean_img.shape, dtype=np.int16)
    noisy_img = np.clip(clean_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    cv2.imwrite(os.path.join(train_good, f"train_{i:03d}.png"), noisy_img)
    if i < 5:
        cv2.imwrite(os.path.join(test_good, f"test_good_{i:03d}.png"), noisy_img)

print("[2/2] Generating defective test samples (scratches & contamination)...")
for i in range(5):
    defective_img = create_base_bottle()
    
    if i % 2 == 0:
        # Red scratch defect
        cv2.line(defective_img, (210 + i*10, 220), (270 + i*10, 280), (0, 0, 220), 4)
    else:
        # Dark contamination spot defect
        cv2.circle(defective_img, (250, 250 + i*15), 18, (20, 20, 20), -1)

    cv2.imwrite(os.path.join(test_defective, f"test_defect_{i:03d}.png"), defective_img)

print("✅ Success! Benchmark dataset generated in './data/bottle'")
