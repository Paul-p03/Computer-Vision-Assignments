
import cv2
import numpy as np
import os

# ========== Setup ==========
# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# ========== Load Image ==========
# Use your own image path here

image = cv2.imread(r'C:\Users\ppatu\Computer-Vision-Assignments\Computer-Vision-Assignments\Assignment4\Image\SiftImage.png')  # Replace with your actual image path
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ========== Apply Transformations ==========
# 1. Scaling
scaled = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

# 2. Rotation
(h, w) = gray.shape
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(gray, M, (w, h))

# 3. Illumination Change (brighter)
illumination = cv2.convertScaleAbs(gray, alpha=1.2, beta=30)

# Save transformed images
cv2.imwrite("output/gray.jpg", gray)
cv2.imwrite("output/scaled.jpg", scaled)
cv2.imwrite("output/rotated.jpg", rotated)
cv2.imwrite("output/illumination.jpg", illumination)

# ========== Initialize SIFT ==========
sift = cv2.SIFT_create()

# ========== Function to Extract and Save Keypoints ==========
def process_image(img, name):
    keypoints, descriptors = sift.detectAndCompute(img, None)

    # Draw rich keypoints to show orientation
    key_img = cv2.drawKeypoints(
        img, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imwrite(f"output/keypoints_{name}.jpg", key_img)
    print(f"[INFO] Processed: {name} | Keypoints: {len(keypoints)}")
    return keypoints, descriptors

# Process original and transformed images
kp_gray, desc_gray = process_image(gray, "gray")
kp_scaled, desc_scaled = process_image(scaled, "scaled")
kp_rotated, desc_rotated = process_image(rotated, "rotated")
kp_illumination, desc_illumination = process_image(illumination, "illumination")

# ========== Optional: Match Keypoints Between Original and Transformed ==========
def match_keypoints(img1, kp1, desc1, img2, kp2, desc2, name):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    # Apply ratio test as per Lowe's paper
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    matched_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    cv2.imwrite(f"output/match_{name}.jpg", matched_img)
    print(f"[INFO] Saved match visualization: match_{name}.jpg")

# Match original vs each transformed version
match_keypoints(gray, kp_gray, desc_gray, scaled, kp_scaled, desc_scaled, "gray_vs_scaled")
match_keypoints(gray, kp_gray, desc_gray, rotated, kp_rotated, desc_rotated, "gray_vs_rotated")
match_keypoints(gray, kp_gray, desc_gray, illumination, kp_illumination, desc_illumination, "gray_vs_illumination")
