import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

img=cv.imread("biju2.png")

gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces=face_cascade.detectMultiScale(gray,1.3, 4)

print(faces)

for x,y,w,h in faces:
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 4)

cv.imshow("Original", img)


cv.waitKey(0)
cv.destroyAllWindows()

'''import cv2 as cv

# Load the Haar Cascade Classifier
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize the camera
camera = cv.VideoCapture(0)  # Use 0 for the default camera; change the index for other cameras

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)

    # Draw rectangles around detected faces
    for x, y, w, h in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Display the resulting frame
    cv.imshow("Real-Time Face Detection", frame)

    # Break the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv.destroyAllWindows()'''
