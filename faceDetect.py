import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
import tensorflow as tf
from model import create_model
from keras import backend as K
from keras.models import Model
from itertools import chain
import dlib
from align import AlignDlib


class Face_detector:
    haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")   # we will load cascade classifier into this variable later
    faces = None # we will store detected faces Coordinates
    hog =   dlib.get_frontal_face_detector()
    alignment = AlignDlib('shape_predictor_68_face_landmarks.dat')

    def __init__(self):
        self.image_path = ""
        self.image = None
        self.grayImage = None
    
    def loadImage(self):
        self.image = cv2.imread(self.image_path) # load image 
        self.grayImage = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY) # convert it to gray as cv2 face detector expecte


    def isFace(self,img,path):
        self.image_path=path
        self.grayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = Face_detector.haar_cascade.detectMultiScale(self.grayImage,scaleFactor=1.1,minNeighbors=5)
        if(len(faces)==1):
            x , y , w , h = faces[0]
            #allign = Face_detector.alignment.align(96, img,  dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h), landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
            cv2.imwrite(self.image_path,img[y:y+h,x:x+w])
            return "Accepted Image"
        elif(len(faces)>1):
            return "many faces!"
        elif (len(faces)==0):
            return "no faces detected.."

    def isFace2(self,img,path):
        self.image_path=path
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #print(Face_detector.alignment.getAllFaceBoundingBoxes(img))
        #print(type(Face_detector.alignment.getAllFaceBoundingBoxes(img)))
        faces = Face_detector.alignment.getAllFaceBoundingBoxes(img)
        if(len(faces)==1):
            x , y , w , h = faces[0].left(), faces[0].top() , faces[0].width(), faces[0].height()
            cv2.imwrite(self.image_path,cv2.resize(img[y:y+h,x:x+w],(96,96)))
            allign = Face_detector.alignment.align(96, img, faces[0], landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
            cv2.imwrite(self.image_path+"all.jpg",allign)
            return "Accepted Image"
        elif(len(faces)>1):
            return "many faces!"
        elif (len(faces)==0):
            return "no faces detected.."
    def align(self,img,x,y,w,h):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        return Face_detector.alignment.align(96, img,  dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h), landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
class decoder:
    nn4_small2_pretrained = create_model()
    nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')
    def __init__(self):
        pass
    
    def decode_images(self,imagesPath,id,):
        print("start decoding new user Images...")
        metaData = 0
        for i,path in enumerate(imagesPath):
            img = cv2.imread(path)
            img = img [...,::-1]
            img = (img / 255.).astype(np.float32)
            metaData+=decoder.nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
        #metaData/=5
        print("finish decoding new user Images...returning..\n")
        return metaData

    def decode_single_image(self,img):
        img = img [...,::-1]
        img = (img / 255.).astype(np.float32)
        img = cv2.resize(img,dsize = (96,96)).reshape(1,96,96,3)
        return decoder.nn4_small2_pretrained.predict(img)[0]
    
    def getIndex(self,current,allUsers):
        distanc = []#np.array(np.sum(np.square(allUsers - current)),ndmin =1)
        for i in range(allUsers.shape[0]):
            d = np.sum(np.square((np.square(allUsers[i]) - np.square(current))))
            distanc.append(d)
        print(len(distanc))
        #list(np.sum(np.square(allUsers - current),axis=1))
        return min(distanc) , np.argmin(distanc)

    def test(self,testImage,referance):
        """imgTest = cv2.imread(testImage).resize(96,96).reshape(1,96,96,3)
        imgTest = decoder.nn4_small2_pretrained.predict(imgTest)[0]
        re = np.load(referance)
        """
        test = decoder.nn4_small2_pretrained.predict(np.expand_dims(testImage, axis=0))[0]
        re =  decoder.nn4_small2_pretrained.predict(np.expand_dims(referance, axis=0))[0] #referance #
        l = np.sum(np.square(test - re))#np.linalg.norm(test-re)
        print(l)

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]
    
"""nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')
haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")   # we will load cascade classifier into this variable later

img1 = load_image("trainingData\\Gerhard_Schroeder\\Gerhard_Schroeder_0007.jpg")
gray = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
f = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
x ,y,w,h=f[0]
img1 = cv2.resize(img1[y:y+h,x:x+h],(96,96))
img1 = (img1 / 255.).astype(np.float32)
emb1 = nn4_small2_pretrained.predict(np.expand_dims(img1, axis=0))[0]

img2 = load_image("trainingData\\Gerhard_Schroeder\\Gerhard_Schroeder_0005.jpg")
gray = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
f = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
x ,y,w,h=f[0]
img2 = cv2.resize(img2[y:y+h,x:x+h],(96,96))
img2 = (img2 / 255.).astype(np.float32)
emb2 = nn4_small2_pretrained.predict(np.expand_dims(img2, axis=0))[0]

print(np.sum(np.square(emb1 - emb2)))
"""
