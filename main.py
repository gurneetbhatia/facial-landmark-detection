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

def get_polygon_area(contour, image):
    xpts = sorted([point[0] for point in contour])
    left_pt = xpts[0] # min
    right_pt = xpts[-1] # max

    ypts = sorted([point[1] for point in contour])
    top_pt = ypts[0] # min
    bottom_pt = ypts[-1] # max

    points = []
    for x in range(left_pt, right_pt):
        for y in range(top_pt, bottom_pt):
            # lies inside the contour or at the edge
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                points.append(image[x, y])
    return np.array(points)

def expand_lips(horizontal_factor, vertical_factor, contour, image):
    xpts = sorted([point[0] for point in contour])
    left_pt = xpts[0] # min
    right_pt = xpts[-1] # max

    ypts = sorted([point[1] for point in contour])
    top_pt = ypts[0] # min
    bottom_pt = ypts[-1] # max

    imape_template = image.copy()
    print(left_pt, right_pt, top_pt, bottom_pt)
    print(image[0][0])
    for x in range(left_pt, right_pt):
        for y in range(top_pt, bottom_pt):
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                print("Colour at ("+str(x)+", "+str(y)+"):", image[y, x])

def find_nearest(array, value):
  idx = np.array([np.linalg.norm(x+y) for (x,y) in array-value]).argmin()
  return idx

def get_lip_point_ratios(contour, image):
    # return the ratio between adjacent lip points on the contour
    ratios = []
    for (index, point) in enumerate(contour):
        if index == len(contour) - 1:
            ratios.append(contour[0]/contour[-1])
            continue
        ratios.append(contour[index]/contour[index+1])
    return ratios

def scale_lips(initial_ratios, final_ratios, contour, image):
    xpts = sorted([point[0] for point in contour])
    left_pt = xpts[0] # min
    right_pt = xpts[-1] # max
    xmid = (left_pt + right_pt)/2

    ypts = sorted([point[1] for point in contour])
    top_pt = ypts[0] # min
    bottom_pt = ypts[-1] # max
    ymid = (top_pt + bottom_pt)/2

    # for horizontal expansion, look at ymid
    # if the x < ymid, expand to the left, else expand to the right
    # for vertical expansion, look at xmid
    # if the y < xmid, expand towards the top, else towards the bottom
    marked_indices = []
    image_template = image.copy()
    for x in range(left_pt, right_pt):
        for y in range(top_pt, bottom_pt):
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                #print("Colour at ("+str(x)+", "+str(y)+"):", image[y, x])
                colour = image[y, x]
                # first find the horizontal expansion
                #print("Closest to ("+str(x)+", "+str(y)+"): ", find_nearest(contour, [x, y]))
                contour_pt = find_nearest(contour, [x, y])
                final_ratio = final_ratios[contour_pt]/initial_ratios[contour_pt]
                print("Final Ratio: ("+str(x)+", "+str(y)+"):", final_ratio)

                # first expand horizontally
                if x < ymid:
                    # expand to the left
                    image_template[y, x-1] = colour
                else:
                    # expand to the right
                    image_template[y, x+1] = colour
                if y < ymid:
                    # expand towards the top
                    image_template[y-1, x] = colour
                else:
                    # expand towards the bottom
                    image_template[y+1, x] = colour
    return image_template


image_data = load_image("image1.png")

# the data for the lips in landmarks is from 50-68th index of landmarks[0][0]
# lips1 = get_lips(image_data1[1])
outer_lip = get_outer_lip(image_data[1])
inner_lip = get_inner_lip(image_data[1])
img = draw_polygon(image_data[2], outer_lip)
img = draw_polygon(img, inner_lip)
print("here", outer_lip[0])
contour = np.int32(outer_lip)
# print(get_polygon_area(contour, image_data[2]))

vertical_factor = 2
horizontal_factor = 1
#expand_lips(horizontal_factor, vertical_factor, contour, image_data[2])
initial_ratios = get_lip_point_ratios(contour, image_data[2])
print(initial_ratios)
final_ratios = list(map(lambda ratio: ratio * 2, initial_ratios))
print(final_ratios)

im = scale_lips(initial_ratios, final_ratios, contour, image_data[2])
plt.axis("off")
plt.imshow(im)
plt.show()
