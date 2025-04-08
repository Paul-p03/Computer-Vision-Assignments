import cv2
import numpy as np
import os

os.makedirs("output", exist_ok=True)

# Load image
img = cv2.imread(r'C:\Users\ppatu\Computer-Vision-Assignments\Computer-Vision-Assignments\Assignment4\Image\SiftImage.png')  # Replace with your actual image path
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("output/original_gray.jpg", gray)

# a. Scaling
scaled_up = cv2.resize(gray, None, fx=1.5, fy=1.5)
scaled_down = cv2.resize(gray, None, fx=0.5, fy=0.5)
cv2.imwrite("output/scaled_up.jpg", scaled_up)
cv2.imwrite("output/scaled_down.jpg", scaled_down)

# b. Rotation
(h,w) = gray.shape
center = (w//2,h//2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(gray, M, (w,h))
cv2.imwrite("output/rotated.jpg", rotated)


# c. Brightness scale
bright = cv2.convertScaleAbs(gray, alpha=1.0, beta=60)  
dark = cv2.convertScaleAbs(gray, alpha=1.0, beta=-60)  
cv2.imwrite("output/bright.jpg", bright)
cv2.imwrite("output/dark.jpg", dark)

# Initialize SIFT detector
sift = cv2.SIFT_create()

def extract_and_save(img, name):
    keypoints, descriptors = sift.detectAndCompute(img, None)
    img_kp = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(f"output/{name}_keypoints.jpg", img_kp)
    print(f"{name} - Keypoints detected: {len(keypoints)}")
    return keypoints, descriptors

# Extract the features from each image
extract_and_save(gray, "original")
extract_and_save(scaled_up, "scaled_up")
extract_and_save(scaled_down, "scaled_down")
extract_and_save(rotated, "rotated")
extract_and_save(bright, "bright")
extract_and_save(dark, "dark")


def match_keypoints(img1, img2, name):
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    matched_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    cv2.imwrite(f"output/match_{name}.jpg", matched_img)
    print(f"Saved match visualization: match_{name}.jpg")

# Matching the identical features between images
match_keypoints(gray, rotated, "original_vs_rotated")
