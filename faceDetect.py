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
#from align import AlignDlib


class Face_detector:
    haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml") 
    faces = None # we will store detected faces Coordinates
    hog =   dlib.get_frontal_face_detector()
    #alignment = AlignDlib('shape_predictor_68_face_landmarks.dat')

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
            cv2.imwrite(self.image_path,img[y:y+h,x:x+w])
            return "Accepted Image"
        elif(len(faces)>1):
            return "many faces!"
        elif (len(faces)==0):
            return "no faces detected.."
    
    
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
            img = (img / 255.).astype(np.float32)
            metaData+=decoder.nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
        print("finish decoding new user Images...returning..\n")
        return metaData

    def decode_single_image(self,img):
        img = (img / 255.).astype(np.float32)
        img = cv2.resize(img,dsize = (96,96)).reshape(1,96,96,3)
        return decoder.nn4_small2_pretrained.predict(img)[0]
    
    def getIndex(self,current,allUsers):
        distanc = [] 
        for i in range(allUsers.shape[0]):
            d = np.sum(np.square((np.square(allUsers[i]) - np.square(current))))
            distanc.append(d)
        print(len(distanc))
        return min(distanc) , np.argmin(distanc)

    def test(self,testImage,referance):

        test = decoder.nn4_small2_pretrained.predict(np.expand_dims(testImage, axis=0))[0]
        re =  decoder.nn4_small2_pretrained.predict(np.expand_dims(referance, axis=0))[0] 
        l = np.sum(np.square(test - re))
        print(l)

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]
    
