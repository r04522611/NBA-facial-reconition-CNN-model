import numpy as np  # import numpy, opencv(cv2),sys
import cv2
import sys
import csv
import numpy as np
import random
#Players' graph name for 10 player (5 graph for each)
Playername = ['JR Smith1.jpg', 'JR Smith2.jpg', 'JR Smith3.jpg','JR Smith4.jpg', 'JR Smith5.jpg',
              'Anthony Davis1.jpg','Anthony Davis2.jpg', 'Anthony Davis3.jpg', 'Anthony Davis4.jpg','Anthony Davis5.jpg',
              'Jeremy Lin1.jpg','Jeremy Lin2.jpg', 'Jeremy Lin3.jpg', 'Jeremy Lin4.jpg','Jeremy Lin5.jpg',
              'Kawhi Leonard1.jpg','Kawhi Leonard2.jpg', 'Kawhi Leonard3.jpg', 'Kawhi Leonard4.jpg','Kawhi Leonard5.jpg',
              'Kevin Love1.jpg','Kevin Love2.jpg', 'Kevin Love3.jpg', 'Kevin Love4.jpg','Kevin Love5.jpg',
              'kobe bryant1.jpg','kobe bryant2.jpg', 'kobe bryant3.jpg', 'kobe bryant4.jpg','kobe bryant5.jpg',
              'LeBron James1.jpg','LeBron James2.jpg', 'LeBron James3.jpg', 'LeBron James4.jpg','LeBron James5.jpg',
              'Russel WestBrook1.jpg','Russel WestBrook2.jpg', 'Russel WestBrook3.jpg', 'Russel WestBrook4.jpg','Russel WestBrook5.jpg'
,              'Stephen Curry1.jpg','Stephen Curry2.jpg', 'Stephen Curry3.jpg', 'Stephen Curry4.jpg','Stephen Curry5.jpg',
              'Tony Parker1.jpg','Tony Parker2.jpg', 'Tony Parker3.jpg', 'Tony Parker4.jpg','Tony Parker5.jpg']

#  10 playername of my favorite 
Player= [ 'JR Smith','JR Smith','JR Smith','JR Smith','JR Smith',
          'Anthony Davis','Anthony Davis','Anthony Davis','Anthony Davis','Anthony Davis',
          'Jeremy Lin','Jeremy Lin', 'Jeremy Lin','Jeremy Lin','Jeremy Lin',
          'Kawhi Leonard','Kawhi Leonard','Kawhi Leonard','Kawhi Leonard','Kawhi Leonard',
          'Kevin Love','Kevin Love', 'Kevin Love', 'Kevin Love','Kevin Love',
          'Kobe Bryant','Kobe Bryant','Kobe Bryant','Kobe Bryant','Kobe Bryant',
          'LeBron James','LeBron James','LeBron James','LeBron James','LeBron James',
          'Russel WestBrook','Russel WestBrook','Russel WestBrook','Russel WestBrook','Russel WestBrook',
          'Stephen Curry','Stephen Curry','Stephen Curry','Stephen Curry','Stephen Curry',
          'Tony Parker', 'Tony Parker', 'Tony Parker', 'Tony Parker','Tony Parker']


#Test graph jpg  & Name
TestName =['Teacher.jpg', 'Michael.jpg', 'Eileen.jpg']
Test = ['Teacher','Michael','Eileen']


#The factial character acquisition function
def CharacterAcqusition(i, outputlist, ReadInJPG, ReadInName):  # define the function of factial character acquisition

    # cv2:cascade classifier for face
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
    # cv2 : cascade classifier for eye
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    # cv2 : cascade classifier fro mouth
    mouth_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
    img = cv2.imread(ReadInJPG[i])
    # import the graph from the list
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert into gray scale for detectation
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    mouth = mouth_cascade.detectMultiScale(gray, 1.3, 5)
    # face and mouth are the class of contain the position(x,y) and dimention
    # (w,h) of target area
    a = 0
    b = 0
    # counter a & b
    for (x, y, w, h) in faces:  # acess the face loaction and dimention
        deltaX = []
        deltaY = []
        deltaEW =[]
        deltaEH = []
        # depicit the area of face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0),2) 
        # region for gray (in eyes dectation)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]  # region for shown the image
        # Sent the gray region for eyes dectation
        eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:  # acess the eyes loaction and dimention
        if ey + eh < y + h and b < 2:  # only executive when the eye location is in face region for both eyes
            cv2.rectangle(roi_color, (ex, ey),(ex + ew, ey + eh), (0, 255, 0), 1)
            deltaX.append(ex)  # record the x pixel of each eye
            deltaY.append(ey)  # record the y pixel of each eye
            deltaEW.append(ew)
            deltaEH.append(eh)
            b += 1  # counter +=1

    for (mx, my, mw, mh) in mouth:  # acess the mouth loaction and dimention
        if w>mw>ew and ey + eh< my+mh and a == 0:  # only executive when the mouth location is in face region for only one mouth
            cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (0, 0, 255), 1)
            a += 1

    # computing the factial reconition character
    
    lengthEs = abs(deltaX[0] - deltaX[1])  # the width between eyes
    # the heigh of mouth to eyes
    lengthEN = int(abs(0.5 * (deltaY[1] - deltaY[0]) + my))
    AreaM = mw * mh  # area of mouth
    AreaF = w * h  # area of face
    LEye = int(deltaEW[0]*deltaEH[0]*100/AreaF)
    REye = int(deltaEW[1]*deltaEH[1]*100/AreaF)
    Mouth = int(AreaM*100/AreaF)
    MandE = LEye+REye+Mouth
    EtoM = int((deltaEW[0]*deltaEH[0]+deltaEW[1]*deltaEH[1])*100/AreaM)
    LEp= int(lengthEs*100/w)
    LEMp = int(lengthEN*100/h)
    Classfication = (i//5)
    #cv2.imshow(Playername[i], img)  # outport the final graph

    Outcome = [ReadInName[i],lengthEs,lengthEN,AreaM,AreaF,LEye,REye,Mouth,MandE,EtoM,LEp,LEMp,Classfication]
    outputlist.append(Outcome)

#================================================== Executive
# print the basic information of factial reconition character

outputList =[['PlayerName', 'lengthEs','lengthEN','AreaM','AreaF','LEye','REye','Mouth','MandE','EtoM','LEp','LEMp','Classfication']]    
#Output the factial reconition character

TrainList50= []
PredictList3= []

wr= open('Train.csv','w',newline='')
ww = csv.writer(wr)
ww.writerows(outputList) #  Weite the first column in it 
for i in range(50):  # executive for 50 times for 50 graphs
    CharacterAcqusition(i, TrainList50, Playername, Player)

#write to the csv file in 5%  normal distribution to duplicate the 50 data for increasing the variability
# (50--> 50 data *7 characteristic * 40 times + 50 original data = 14050 )
#output to the Test.csv
for i in TrainList50:  

    ramdomlist = []
    InputItem=[]
    outputlist = [] 
    for y in i:
        InputItem.append(y)   
    for s in range (0, 280):
        ramdomlist.append(np.random.normal(0, 0.5))
    for w in range (0,7):
        for j in range(0,40):
            outputlist.append(i[0:w+5])
            outputlist[40*w+j].append(round(InputItem[w+5]*(1+0.05*(ramdomlist[40*w+j])),3))
            for h in i[w+6:13]:
                outputlist[40*w+j].append(h)
    outputlist.append(i[0:13])
    ww.writerows(outputlist)
    print(i)
wr.close()


#open a new Test.csv
# write the 50 original data & 3 Predict data in it
TestW= open('Test.csv','w',newline='')
Testwrite = csv.writer(TestW)
Testwrite.writerows(outputList)

for i in range (3):
    CharacterAcqusition(i, PredictList3, TestName, Test)
for i in TrainList50:
    d=[]
    for y in i:
        d.append(y)
    kk =[]
    kk.append(i)
    Testwrite.writerows(kk)
    
for i in PredictList3:
    d=[]
    for y in i:
        d.append(y)
    kk =[]
    kk.append(i)
    Testwrite.writerows(kk)
TestW.close()
