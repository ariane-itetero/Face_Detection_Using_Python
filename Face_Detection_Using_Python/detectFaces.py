import os
import cv2

# Load the image
img = cv2.imread("group-of-people-1.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(
    "/Users/Ariane/Documents/CODES/Python/Face_Detection_Using_Python/haarcascade_frontalface_default.xml")

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Create a directory to save the cropped images
if not os.path.exists("cropped_faces"):
    os.makedirs("cropped_faces")

# Crop and save each face
for i, (x, y, w, h) in enumerate(faces):
    face_img = img[y:y+h, x:x+w]
    # increase the size of the face
    resized_face = cv2.resize(face_img, (2*w, 2*h))
    cv2.imwrite(f"cropped_faces/face_{i}.jpg", resized_face)
