import pandas as pd
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from faceDetect import Face_detector as fc

def creatData():
    if not os.path.exists("attendance-data"):
        os.makedirs("attendance-data")
    data = {"ID":[],"Name":[],"Path":[]}
    if not os.path.isfile("attendance-data\\data.csv"):
        df = pd.DataFrame(data = data,columns=['ID','Name','Path'])
        df.to_csv("attendance-data\\data.csv",index=False)


def appendData(id,name,path):
    data = {'ID':id,'Name':name,'Path':path}
    df = pd.read_csv("attendance-data\\data.csv")
    df.loc[id-1] = data
    df.to_csv("attendance-data\\data.csv",index=False)


def takePic(name):
    df = pd.read_csv("attendance-data\\data.csv")
    id = len(df["ID"]) + 1
    path = 'attendance-data\\{} - {}'.format(id,name)
    if not os.path.exists(path):
        os.makedirs('attendance-data\\{} - {}'.format(id,name))
    print("Please take 5 photos!, press Enter to take photo")
    cam = cv2.VideoCapture(0)
    imageCounter = 0
    while(True):
        ret,frame = cam.read()
        cv2.imshow("Take Photos",frame)

        if not ret :
            print("error in take photos!")
            break
        key = cv2.waitKey(1)
        if imageCounter==5:
            print("thank you your 5 photos saved!")
            break
        elif key%256 == 13:
            imgName = "{}-pic-{}.png".format(name,imageCounter+1)
            cv2.imwrite(os.path.join(path,imgName),frame)
            isFace = faceCheck.isFace(os.path.join(path,imgName))
            if(isFace=="Accepted Image"):
                imageCounter+=1
                print("image {} saved".format(imgName))
            else:
                #os.remove(os.path.join(path,imgName))
                print(isFace,"please try again!")
            
    cam.release()
    cv2.destroyAllWindows()
    return id,path

def showAllData():
    fileName = "data.csv"
    filePath = "attendance-data\\"
    df = pd.read_csv(filePath+fileName,index_col=0)
    if(len(df)==0):
        print("there are no data to show please enter some users first!")
        exit(0)
    print(df)
    print("\n..................................................\n")


def showUserPic(id):

    fileName = "attendance-data\\data.csv"
    df = pd.read_csv(fileName)
    if(len(df)==0):
        print("there are no data to show please enter some users first!")
        exit(0)
    allIds = df['ID'].tolist()
    while(id not in allIds):
        id  = int(input("ID: {} not exist, please enter valid ID or 0 to exit: ".format(id)))
        if(not id):
            print("exit..")
            exit(0)
    df.set_index("ID",inplace=True)
    imagesPath = df.loc[(id),"Path"]
    pics = os.listdir(imagesPath)
    fig , axeslist = plt.subplots(ncols = 3 , nrows=2,figsize=(15,10))
    for ind,i in enumerate(pics):
        img = mpimg.imread(imagesPath+'\\'+i)
        axeslist.ravel()[ind].imshow(img)
    plt.show()
    print("\n..................................................\n")


def addNewUser():
    name = str(input("Enter new user name : "))
    id,path=takePic(name)
    appendData(id,name,path)
    print("new user added !\nID:{}, name:{}".format(id,name))
    print("\n..................................................\n")


def main():
    creatData()
    while(True):
        task = int(input("enter num of what you want to do?\n1)Add new use ?,  2)show All data ? , 3)plot user pic? , , 4)quit : "))
        while(task>4) or task<0:
            task = int(input("wrong num! please enter num of option you want to do \n1)Add new use ?,  2)show All data ? , 3)plot user pic? , 4)quit "))
        if(task==1):
            addNewUser()
        elif(task==2):
            showAllData()
        elif(task==3):
            id = int(input("enter user Id: "))
            showUserPic(id)
        else:
            exit(0)


if __name__== "__main__":
    faceCheck = fc() # declare global opject of face detector class  
    main()