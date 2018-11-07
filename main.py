import pandas as pd
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from faceDetect import Face_detector as fc
import shutil


def creatData():
    """
    this function create our data folder and csv file if they are not exist
    """


    if not os.path.exists("attendance-data"):
        os.makedirs("attendance-data")
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
    while(True):
        # initalize cv2 to take pic
        ret,frame = cam.read()
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
            imgName = "{}-pic-{}.png".format(name,imageCounter+1) # generate image name using it's number 
            cv2.imwrite(os.path.join(path,imgName),frame) # save image in user folder 
            isFace = faceCheck.isFace(os.path.join(path,imgName)) # check if there are faces in pic (object return [accepted image or many faces or no faces detcted])
            if(isFace=="Accepted Image"): # accept pic if there are only one face in it 
                imageCounter+=1
                print("image {} saved".format(imgName)) # save accepted image 
            else:
                print(isFace,"please try again!, or enter 0 to quit")

        elif key%256 == 48: # user press 0 
            print("WARNING: all new user data will REMOVE!, press d to quit.")
        
        elif key == ord('d'): # delete pic and exit 
            print("delete data and exit ...")
            shutil.rmtree(path)
            exit(0)
    # end cam process   
    cam.release() 
    cv2.destroyAllWindows()
    return id,path # return generated id and image folder path

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
    name = str(input("Enter new user name : ")) # get new user name 
    id,path=takePic(name)
    appendData(id,name,path)
    print("new user added !\nID:{}, name:{}".format(id,name))
    print("\n..................................................\n")


def main():
    creatData()
    while(True):
        task = (input("enter num of what you want to do?\n1)Add new use ?,  2)show All data ? , 3)plot user pic? , 4)quit : "))
        if(task=='1'):
            addNewUser()
        elif(task=='2'):
            showAllData()
        elif(task=='3'):
            id = int(input("enter user Id: "))
            showUserPic(id)
        elif (task=='4'):
            print("exit..")
            exit(0)
        else:
            task = (input("wrong num! please enter num of option you want to do \n1)Add new use ?,  2)show All data ? , 3)plot user pic? , 4)quit : "))


if __name__== "__main__":
    faceCheck = fc() # declare global object of face detector class  
    main()