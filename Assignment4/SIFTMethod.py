import cv2 

# reading the image
img = cv2.imread(r'C:\Users\ppatu\Computer-Vision-Assignments\Computer-Vision-Assignments\Assignment4\Image\SiftImage.png')
# convert to greyscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create SIFT feature extractor
sift = cv2.SIFT_create()

# detect features from the image
keypoints = sift.detect(img)

# draw the detected key points
sift_image = cv2.drawKeypoints(gray, keypoints, img)
# show the image
cv2.imshow('image', sift_image)
# save the image
cv2.imwrite("table-sift.jpg", sift_image)
cv2.waitKey(0)
cv2.destroyAllWindows()