import pandas as pd
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from faceDetect import Face_detector as fc
from faceDetect import decoder 
import shutil


def creatData():
    """
    this function create our data folder and csv file if they are not exist
    """


    if not os.path.exists("attendance-data"):
        os.makedirs("attendance-data")
    if not os.path.exists("attendance-data\metaData"):
        os.makedirs("attendance-data\metaData")
    data = {"ID":[],"Name":[],"Path":[]}
    if not os.path.isfile("attendance-data\\data.csv"):
        df = pd.DataFrame(data = data,columns=['ID','Name','Path'])
        df.to_csv("attendance-data\\data.csv",index=False)


def appendData(id,name,path):
    """
    this function append new user data into our csv file 

    args:
    (input)
        id: new user id 
        name : new user name
        path : path to new user images
    """
    data = {'ID':id,'Name':name,'Path':path}
    df = pd.read_csv("attendance-data\\data.csv")
    df.loc[id-1] = data
    df.to_csv("attendance-data\\data.csv",index=False)


def takePic(name):
    """
    this function create id for new user then take 5 pic for him and make sure that every pic has one face 
    args:
    (input)
        name : new user name (to name user image floder )
    """

    # read csv file and git last id to generate new one 
    df = pd.read_csv("attendance-data\\data.csv")
    id = len(df["ID"]) + 1

    # create folder for user images using his id and name to be distinct name 
    path = 'attendance-data\\{} - {}'.format(id,name)
    if not os.path.exists(path): # make sure that file not exist before 
        os.makedirs('attendance-data\\{} - {}'.format(id,name))
    
    # start taking pic of new user 
    print("Please take 5 photos!, press Enter to take photo or 0 to quit")
    cam = cv2.VideoCapture(0)
    imageCounter = 0
    imagesNames=[]
    #path2 = input("Enter Image Path")
    #frame = cv2.imread(path2)
    imgName = "{}-pic-{}.png".format(name,0)
    #isFace = faceCheck.isFace(frame,os.path.join(path,imgName))
    imagesNames.append(os.path.join(path,imgName))
    q = False
    while(True):
        # initalize cv2 to take pic
        ret,frame = cam.read()
        grayImage = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = faceCheck.haar_cascade.detectMultiScale(grayImage,scaleFactor=1.1,minNeighbors=5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
        cv2.imshow("Take Photos",frame)
        
        #make sure that cam work
        if not ret :
            print("error in take photos!")
            break
        # init wait key
        key = cv2.waitKey(1) & 0xFF
        # break if 5 pics saved 
        if imageCounter==5:
            print("thank you your 5 photos saved!")
            break
        # wait for user to press ENTER or 0 or d key 
        elif key%256 == 13:
            imgName = "{}-pic-{}.png".format(name,imageCounter) # generate image name using it's number 
            imagesNames.append(os.path.join(path,imgName))
            #cv2.imwrite(os.path.join(path,imgName),frame) # save image in user folder 
            isFace = faceCheck.isFace(frame,os.path.join(path,imgName)) # check if there are faces in pic (object return [accepted image or many faces or no faces detcted])
            if(isFace=="Accepted Image"): # accept pic if there are only one face in it 
                imageCounter+=1
                print("image {} saved".format(imgName)) # save accepted image 
            else:
                print(isFace,"please try again!, or enter 0 to quit")

        elif key%256 == 48: # user press 0 
            q = True
            print("WARNING: all new user data will REMOVE!, press d to quit.")
        
        elif key == ord('d') and q: # delete pic and exit 
            print("delete data and exit ...")
            shutil.rmtree(path)
            exit(0)
    # end cam process   
    cam.release() 
    cv2.destroyAllWindows()
    
    return id,path,imagesNames# return generated id and image folder path

def showAllData():
    """
    this function print all user data stored in our csv file
    """
    fileName = "data.csv"
    filePath = "attendance-data\\"
    df = pd.read_csv(filePath+fileName,index_col=0)
    if(len(df)==0): # if file is empty
        print("there are no data to show please enter some users first!")
        print("\n..................................................\n")
        return
    print(df)
    print("\n..................................................\n")

def showUserPic(id):

    """
    this function plot 5 pic of user with the givin id 
    args:
    (input)
        id: id of user who want to plot his images
    """
    fileName = "attendance-data\\data.csv"
    df = pd.read_csv(fileName)
    if(len(df)==0): # no data exist yet 
        print("there are no data to show please enter some users first!")
        print("\n..................................................\n")
        return
    # check if given id is valis 
    allIds = df['ID'].tolist() # make list of all id in our csv file 
    while(id not in allIds):
        id  = int(input("ID: {} not exist, please enter valid ID or 0 to back : ".format(id)))
        if(not id):
            print("back to main menu..")
            print("\n..................................................\n")
            return
    # get user images
    df.set_index("ID",inplace=True) # make file indexed by user ids
    imagesPath = df.loc[(id),"Path"] # git images folder path of given id 
    pics = os.listdir(imagesPath) # list names of images 
    fig , axeslist = plt.subplots(ncols = 3 , nrows=2,figsize=(15,10)) # init subplots with 3 X 2 
    for ind,i in enumerate(pics): #iterate on user images 
        img = mpimg.imread(imagesPath+'\\'+i) # load image 
        axeslist.ravel()[ind].imshow(img) # set it on out subplots
    plt.show() # plot our subplot after load all user image on it 
    print("\n..................................................\n")


def addNewUser():
    """
    this function handle the hole process of adding new user int our data, calling helper function like (takePic,appendData)
    """
    first = True
    if os.path.exists("attendance-data\\metaData\\allMeta.npy"):
        first = False
        allMetaData = np.load("attendance-data\\metaData\\allMeta.npy")
    else:
        allMetaData = np.zeros((1,128))
    name = str(input("Enter new user name : ")) # get new user name 
    id,path,imagesNames=takePic(name)
    appendData(id,name,path)

    print(imagesNames[0])
    test = cv2.imread(imagesNames[0],1)
    print(test)
    newMeta = np.array(features.decode_single_image(test)).reshape((1,128))
    print("meta shape is : ",newMeta.shape)
    if not first:
        print("new meta: {}, all meta : {}".format(newMeta.shape,allMetaData.shape))
        allMetaData =  np.vstack([allMetaData,newMeta])
    else:
        allMetaData = newMeta
    #allMetaData.append(newMeta)
    del newMeta
    np.save("attendance-data\\metaData\\allMeta",np.array(allMetaData))
    del allMetaData
    print("new user added !\nID: {}, name:{}".format(id,name))
    print("\n..................................................\n")

def predict():
    currentMetaData = np.load("attendance-data\\metaData\\allMeta.npy")
    currentUserInfo = pd.read_csv("attendance-data\\data.csv")
    names = currentUserInfo['Name']
    cam = cv2.VideoCapture(0)
    cam.get(cv2.CAP_PROP_FPS)
    while(True):
        # initalize cv2 to take pic
        ret,frame = cam.read()
        grayImage = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = faceCheck.haar_cascade.detectMultiScale(grayImage,scaleFactor=1.1,minNeighbors=5)
        if(len(faces)!=0):
            for (x,y,w,h) in faces:
                dist = 10
                name = ""
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
                currentFrame = np.array(features.decode_single_image(frame[y:y+h,x:x+w]).reshape((1,128)))
                dist , ind = features.getIndex(currentFrame,currentMetaData)
                font = cv2.FONT_HERSHEY_SIMPLEX
                if(dist<0.5):
                    name = names[ind]
                else:
                    name = "Nan!!"
                print(dist)
                cv2.putText(frame, name, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
        #make sure that cam work
        cv2.imshow("Take Photos",frame)
        if not ret :
            print("error in take photos!")
            break
        # init wait key
        key = cv2.waitKey(1) & 0xFF
               
        if key == ord('q') : # delete pic and exit 
            print("exit ...")
            break
    # end cam process   
    cam.release() 
    cv2.destroyAllWindows()


def main():
    creatData()
    while(True):
        task=""
        task = (input("enter num of what you want to do?\n1)Add new use ?,  2)show All data ? , 3)plot user pic? , 4)quit : "))
        if(task=='1'):
            addNewUser()
            a = np.load("attendance-data\\metaData\\allMeta.npy")
            print(a.shape)
        elif(task=='2'):
            showAllData()
        elif(task=='3'):
            id = int(input("enter user Id: "))
            showUserPic(id)
        elif (task=='4'):
            print("exit..")
            exit(0)
        elif(task=='5'):
            """test = "attendance-data\\2 - ahmed2\\ahmed2-pic-2.png"
            re = "attendance-data\\metaData\\2.npy"
            features.test(test,re)"""
            predict()

        else:
            task = (input("wrong num! please enter num of option you want to do \n1)Add new use ?,  2)show All data ? , 3)plot user pic? , 4)quit : "))
        

if __name__== "__main__":
    faceCheck = fc() # declare global object of face detector class  
    features = decoder()
    main()
    #img = cv2.imread("trainingData\\Gerhard_Schroeder\\Gerhard_Schroeder_0007.jpg")
    """
    img = cv2.imread("attendance-data\\2 - ayman\\ayman-pic-4.png")
    grayImage = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCheck.haar_cascade.detectMultiScale(grayImage,scaleFactor=1.1,minNeighbors=5)
    x,y,w,h = faces[0]
    img = cv2.resize( img[y:y+h,x:x+w],(96,96))
    cv2.imshow('image1',img)
    img= img[...,::-1]/255
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    img2 = cv2.imread("E:\\me\\my pic\\DSCN3534.jpg")
    grayImage = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    faces = faceCheck.haar_cascade.detectMultiScale(grayImage,scaleFactor=1.1,minNeighbors=5)
    x,y,w,h = faces[0]
    img2 = cv2.resize( img2[y:y+h,x:x+w],(96,96))
    cv2.imshow('image2',img2)
    img2 = img2[...,::-1]/255
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    #img2 = np.load("attendance-data\\metaData\\allMeta.npy")

    #img = cv2.imread("attendance-data\\trainingData\Ariel_Sharon\\Ariel_Sharon_0001.jpg")
    #img2 = cv2.imread("attendance-data\\trainingData\Ariel_Sharon\\Ariel_Sharon_0003.jpg")
    #img2 = np.load("attendance-data\\metaData\\allMeta.npy")
    features.test(img,img2)
    """
a = np.load("attendance-data\\metaData\\allMeta.npy")
print(a.shape)
"""