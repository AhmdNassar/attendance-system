import matplotlib.pyplot as plt
import cv2

class Face_detector:
    haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")   # we will load cascade classifier into this variable later
    faces = None # we will store detected faces Coordinates

    def __init__(self):
        self.image_path = ""
        self.image = None

    def loadImage(self):
        self.image = cv2.imread(self.image_path,0) # load image 
        #self.image = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY) # convert it to gray as cv2 face detector expecte


    def isFace(self,imagePath):
        self.image_path = imagePath
        self.loadImage()
        faces = Face_detector.haar_cascade.detectMultiScale(self.image,scaleFactor=1.1,minNeighbors=5)
        if(len(faces)==1):
            return "Accepted Image"
        elif(len(faces)>1):
            return "many faces!"
        elif (len(faces)==0):
            return "no faces detected.."
