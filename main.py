import cv2
import urllib.request as urlreq
import os
import matplotlib.pyplot as plt
from pylab import rcParams

def detect_faces(image):
    # save the face detection algorithm's url in a variable
    haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
    # save dace detection algorithm's name as haarcascade
    haarcascade = "haarcascade_frontalface_alt2.xml"
    # chech if file is in working directory
    if (haarcascade in os.listdir(os.curdir)):
        print("File exists")
    else:
        # download file from url and save locally as haarcascade_frontalface_alt2.xml, < 1MB
        urlreq.urlretrieve(haarcascade_url, haarcascade)
        print("File downloaded")
    # create an instance of the Face Detection Cascade Classifier
    detector = cv2.CascadeClassifier(haarcascade)
    # use the classifier to detect faces in image_gray
    faces = detector.detectMultiScale(image)
    return faces

def detect_landmarks(faces, image):
    # save facial landmark detection model's url in LBFmodel_url variable
    LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

    # save facial landmark detection model's name as LBFmodel
    LBFmodel = "lbfmodel.yaml"

    # check if file is in working directory
    if (LBFmodel in os.listdir(os.curdir)):
        print("File exists")
    else:
        # download picture from url and save locally as lbfmodel.yaml, < 54MB
        urlreq.urlretrieve(LBFmodel_url, LBFmodel)
        print("File downloaded")

    # create an instance of the Facial Landmark Detector with the model
    landmark_detector  = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel(LBFmodel)

    # Detect landmarks on "image_gray"
    _, landmarks = landmark_detector.fit(image, faces)
    return landmarks

# load the image
pic = "image.jpeg"
image = cv2.imread(pic)

# image processing
# convert the image to RGB colour
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# set the dimensions for cropping the image_rgb
x, y, width, depth = 450, 200, 125, 175
image_cropped = image_rgb[y:(y+depth), x:(x+width)]
# create a copy of the image
image_template = image_cropped.copy()
# create a grayscale version of the cropped image
image_gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)

faces = detect_faces(image_gray)

landmarks = detect_landmarks(faces, image_gray)

for landmark in landmarks:
    for x, y in landmark[0]:
        cv2.circle(image_template, (x, y), 1, (50, 255, 60), 1)
plt.axis("off")
plt.imshow(image_template)
plt.show()
