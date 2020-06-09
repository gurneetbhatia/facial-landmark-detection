import cv2
import urllib.request as urlreq
import os
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np

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

def load_image(url):
    # load the image
    pic = url
    image = cv2.imread(pic)

    # image processing
    # convert the image to RGB colour
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # set the dimensions for cropping the image_rgb
    #x, y, width, depth = 450, 200, 125, 175
    #image_cropped = image_rgb[y:(y+depth), x:(x+width)]
    image_cropped = image_rgb.copy()
    # create a copy of the image
    image_template = image_cropped.copy()
    # create a grayscale version of the cropped image
    image_gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)

    faces = detect_faces(image_gray)

    landmarks = detect_landmarks(faces, image_gray)
    print(landmarks[0][0][0])
    for landmark in landmarks:
        for x, y in landmark[0]:
            cv2.circle(image_template, (x, y), 1, (50, 255, 60), 1)
    plt.axis("off")
    plt.imshow(image_template)
    return faces, landmarks, image_template

def get_outer_lip(landmarks):
    return landmarks[0][0][48:60]

def get_inner_lip(landmarks):
    return landmarks[0][0][60:]

def draw_polygon(image, points):
    '''for (index, point) in enumerate(points[:-1]):
        current_x, current_y = point
        next_x, next_y = points[index+1]
        cv2.line(image, (current_x, current_y), (next_x, next_y), (0, 255, 0), 2)'''
    '''pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(image, np.int32([points]), 1, (0,255,255))
    #cv2.polylines(image,[pts],True,(0,255,255))
    #cv2.polylines(image, points, True, (0, 255, 255))
    plt.axis("off")
    plt.imshow(image)'''
    contours = np.int32([points])
    for cnt in contours:
        cv2.drawContours(image,[cnt],0,(255,255,255),1)

    plt.axis("off")
    plt.imshow(image)
    return image



image_data = load_image("image1.png")

# the data for the lips in landmarks is from 50-68th index of landmarks[0][0]
# lips1 = get_lips(image_data1[1])
outer_lip = get_outer_lip(image_data[1])
inner_lip = get_inner_lip(image_data[1])
img = draw_polygon(image_data2[2], outer_lip)
img = draw_polygon(img, inner_lip)
plt.show()
