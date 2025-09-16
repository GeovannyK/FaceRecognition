"""
Face detection using haar feature-based cascade classifiers
"""

import cv2
import numpy as np
import os

# Training Data
# there is no label 0 / label 0 is empty
subjects = ["", "Alfredo", "Elvis", "Trump"]

# Function to detect face using OpenCV
def detect_face(img):
    # Convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load cascade classifiers:
    face_cascade = cv2.CascadeClassifier("FaceRecognition/haarcascade_frontalface_alt.xml")

    # Detect faces:
    faces = face_cascade.detectMultiScale(gray)
    
    # If no faces are detected then return original img
    if len(faces) == 0:
        return None, None
    
    # Under the assumption that there will be only one face, extract the face area
    (x, y, w, h) = faces[0]
    
    # Return only the face part of the image
    return gray[y:y+h, x:x+w], faces[0]

# Function to read all persons' training images, detect face from each image
# and will return two lists of exactly same size, one list 
# of faces and another list of labels for each face
def prepare_training_data(data_folder_path):
    # ------STEP-1--------
    # Get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
    
    # List to hold all subject faces
    faces = []
    # List to hold labels for all subjects
    labels = []
    
    # Let's go through each directory and read images within it
    for dir_name in dirs:
        # Extract label number of subject from dir_name
        label = int(dir_name)
        print('label', label)
        
        # Build path of directory containing images for current subject
        # sample subject_dir_path = "training-data/1"
        subject_dir_path = os.path.join(data_folder_path, dir_name)
        print('Dir: ', subject_dir_path)
        
        # Get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
        
        # Go through each image name, read image, 
        # detect face and add face to list of faces
        for image_name in subject_images_names:
            # Ignore system files like .DS_Store
            if image_name.startswith("."):
                continue
            
            # Sample image path = training-data/1/1.pgm
            image_path = os.path.join(subject_dir_path, image_name)

            # Read image
            image = cv2.imread(image_path)
            
            # Detect face
            face, rect = detect_face(image)

            # Display an image window to show the image
            draw_rectangle(image, rect)
            cv2.imshow("Training on image...", image)
            cv2.waitKey(100)
            
            # ------STEP-4--------
            # We will ignore faces that are not detected
            if face is not None:
                # Add face to list of faces
                faces.append(face)
                # Add label for this face
                labels.append(label)
                print('Ok ', image_name)
            
        print('Images: ', subject_images_names)
        print(' ')
        
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
    return faces, labels

def draw_rectangle(img, rect):
    if rect is not None:
        (x, y, w, h) = rect
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# This function recognizes the person in image passed
# and draws a rectangle around detected face with name of the 
# subject
def predict(test_img):
    if test_img is None:
        print("Error: Test image not loaded properly.")
        return None

    # Make a copy of the image as we don't want to change original image
    img = test_img.copy()
    # Detect face from the image
    face, rect = detect_face(img)

    if face is not None:
        # Predict the image using our face recognizer 
        label, confidence = face_recognizer.predict(face)
        # Get name of respective label returned by face recognizer
        text = subjects[label]
    
        # Draw a rectangle around face detected
        draw_rectangle(img, rect)
        # Draw name of predicted person
        cv2.putText(img, text, (rect[0], rect[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    
    return img

# Let's first prepare our training data
# Data will be in two lists of same size
# One list will contain all the faces
# and other list will contain respective labels for each face
print("Preparing data...")
faces, labels = prepare_training_data("FaceRecognition/training-data")
print("Data prepared")

# Print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

# Create our LBPH face recognizer
try:
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    face_recognizer = cv2.face_LBPHFaceRecognizer.create()

# Or use EigenFaceRecognizer by replacing above line with 
# face_recognizer = cv2.face.EigenFaceRecognizer_create()

# Or use FisherFaceRecognizer by replacing above line with 
# face_recognizer = cv2.face.FisherFaceRecognizer_create()

# Train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))

# Now that we have the prediction function well defined, next step is to actually call this function on our test images and display those test images to see if our face recognizer correctly recognized them. So let's do it. This is what we have been waiting for. 
print("Predicting images...")

# Load test images
test_img1 = cv2.imread("FaceRecognition/test-data/test1.jpg")
test_img2 = cv2.imread("FaceRecognition/test-data/test2.jpg")
test_img3 = cv2.imread("FaceRecognition/test-data/test3.jpg")

# Check if test images are loaded properly
if test_img1 is None:
    print("Error: Test image 1 not loaded.")
if test_img2 is None:
    print("Error: Test image 2 not loaded.")
if test_img3 is None:
    print("Error: Test image 3 not loaded.")

# Perform a prediction
predicted_img1 = predict(test_img1) if test_img1 is not None else None
predicted_img2 = predict(test_img2) if test_img2 is not None else None
predicted_img3 = predict(test_img3) if test_img3 is not None else None

print("Prediction complete")

# Display images if they are not None
if predicted_img1 is not None:
    cv2.imshow("Img 1", predicted_img1)
if predicted_img2 is not None:
    cv2.imshow("Img 2", predicted_img2)
if predicted_img3 is not None:
    cv2.imshow("Img 3", predicted_img3)

cv2.waitKey(0)
cv2.destroyAllWindows()
