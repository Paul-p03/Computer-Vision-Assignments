import cv2
import numpy as np
import os

#Create output folder
os.makedirs("output", exist_ok=True)

image = cv2.imread(r'C:\Users\ppatu\Computer-Vision-Assignments\Computer-Vision-Assignments\Assignment4\Image\SiftImage.png')  # Replace with your actual image path
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1. Scales image size
scaled = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

# 2. Rotates image
(h, w) = gray.shape
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(gray, M, (w, h))

# 3. Illuminates image
illumination = cv2.convertScaleAbs(gray, alpha=1.2, beta=30)

# Save Transformed Images
cv2.imwrite("output/gray.jpg", gray)
cv2.imwrite("output/scaled.jpg", scaled)
cv2.imwrite("output/rotated.jpg", rotated)
cv2.imwrite("output/illumination.jpg", illumination)

#initializing SIFT
sift = cv2.SIFT_create()

def process_image(img, name):
    keypoints, descriptors = sift.detectAndCompute(img, None)

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


def match_keypoints(img1, kp1, desc1, img2, kp2, desc2, name):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    matched_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    cv2.imwrite(f"output/match_{name}.jpg", matched_img)
    print(f"[INFO] Saved match visualization: match_{name}.jpg")

# Display the matching features between original and transformed images
match_keypoints(gray, kp_gray, desc_gray, scaled, kp_scaled, desc_scaled, "gray_vs_scaled")
match_keypoints(gray, kp_gray, desc_gray, rotated, kp_rotated, desc_rotated, "gray_vs_rotated")
match_keypoints(gray, kp_gray, desc_gray, illumination, kp_illumination, desc_illumination, "gray_vs_illumination")
