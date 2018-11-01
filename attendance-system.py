import pandas as pd
import cv2
import os
import numpy as np

def creatData():
    if not os.path.exists("attendance-data"):
        os.makedirs("attendance-data")
    data = {"ID":[],"Name":[],"Path":[]}
    if not os.path.isfile("attendance-data\\data.csv"):
        df = pd.DataFrame(data = data,columns=['ID','Name','Path'])
        df.to_csv("attendance-data\\data.csv")


def appendData(name,path):
    df = pd.read_csv("attendance-data\\data.csv",index_col=0)
    newID = len(df["ID"]) + 1
    data = {'ID':newID,'Name':name,'Path':path}
    df.loc[newID-1] = data
    df.to_csv("attendance-data\\data.csv")

def takePic(name,id):
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
            print("image {} saved".format(imgName))
            imageCounter+=1
    cam.release()
    cv2.destroyAllWindows()
    return path

