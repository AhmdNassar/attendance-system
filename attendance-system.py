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

takePic("ahmed","1")


"""cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF!=255:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
"""
"""
cam = cv2.VideoCapture(0)

cv2.namedWindow("FivePhotos")

img_counter = 1

while True:
    ret, frame = cam.read()
    cv2.imshow("FivePhotos", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if img_counter == 6:
        print("Thank's for your time.")
        break
    elif k%256 == 32: #Space
        img_name = "opencv_frame_{}.png".format(img_counter)
        path = 'Your Directory Path'+str(num)
        #cv2.imwrite(img_name, frame)
        createfolder(str(num))
        cv2.imwrite(os.path.join(path , img_name), frame)
        #print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()"""