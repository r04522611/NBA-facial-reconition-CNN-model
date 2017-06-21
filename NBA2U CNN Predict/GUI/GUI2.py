from __future__ import print_function 
import theano
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
sgd = SGD(lr=0.01,momentum=0.0,decay=0.0,nesterov=False)
from keras.utils import np_utils
import numpy as np
import csv
import cv2
import os
import random
keras.backend = 'theano'
import tkinter as tk
from tkinter import filedialog as dia
from PIL import Image,ImageTk
from keras.models import model_from_json
PredictInput = []
NBA = {'0':'JR Smith.jpg', '1':'Anthony Davis.jpg','2':'Jeremy Lin.jpg','3':'Kawhi Leonard.jpg',
       '4':'Kevin Love.jpg','5':'Kobe Bryant.jpg','6':'LeBron James.jpg','7':'Russel WestBrook.jpg',
       '8':'Stephen Curry.jpg','9':'Tony Parker.jpg'}   

#Import the grapg from the file
#Use the openCV to acqusite the picture's facial characteristic
def clickImport():
    global options
    options['title'] = "dia.askopenfilename"
    Opemfilename.append(dia.askopenfilename(**options))
    create_image_label()
    CharacterAcqusition (Opemfilename)

# present the load picture in the window    
def create_image_label():
    global i 
    if i >-1:
        global image_file, im, image_label    
        image_file = Image.open(Opemfilename[i])
        im = ImageTk.PhotoImage(image_file)
        image_label = tk.Label(image_frame,image = im)
        image_label.grid(row = 4, column = 0, sticky = 'NW', pady = 8, padx = 20)
        i+=1

        
def CharacterAcqusition(ExamPath):  # define the function of factial character acquisition
    path = str(ExamPath[-1])
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    mouth = mouth_cascade.detectMultiScale(gray, 1.3, 5)
    a = 0
    b = 0
    for (x, y, w, h) in faces:  # acess the face loaction and dimention
        deltaX = []
        deltaY = []
        deltaEW =[]
        deltaEH = []
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0),2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:  
        if ey + eh < y + h and b < 2:  
            cv2.rectangle(roi_color, (ex, ey),
                          (ex + ew, ey + eh), (0, 255, 0), 1)
            deltaX.append(ex) 
            deltaY.append(ey)  
            deltaEW.append(ew)
            deltaEH.append(eh)
            b += 1

    for (mx, my, mw, mh) in mouth:  
        if w>mw>ew and ey + eh< my+mh and a == 0: 
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

    #cv2.imshow('Input Picture', img)  # outport the final graph
    
    global PredictInput
    PredictInput= np.append([], [LEye,REye,Mouth,MandE,EtoM,LEp,LEMp], axis = 0)
    PredictInput = PredictInput.astype('float32')


def compute():
    batch_size = 128
    num_classes = 10
    epochs = 20 

    MyTEST = np.genfromtxt('Test.csv',delimiter = ',',skip_header = 1)
    X_predict =  MyTEST [50:,5:12]
    X_predict =X_predict.astype('float32')
    M_predict= np.row_stack((X_predict,PredictInput)) 

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
     
    # evaluate loaded model on test data
    
    Predict= loaded_model.predict(M_predict,batch_size=128)
    global PlayerPredict
    PlayerPredict= []
    outcome = np.argmax(Predict, axis=1) 
    for i in outcome:
        PlayerPredict.append(NBA[str(i)])


# Load the CNN model and Compute the Predict outcome & Shown the correspond Picture
def CNN ():
    compute()
    img = cv2.imread(PlayerPredict[3])
    cv2.imshow('Predict Outcome', img)  

#Excution line    
#=======================================================================
#Built a window (NBA2U) and 2 buttons("Load Picture"& "Predict (CNN)" )     
win=tk.Tk()
win.title("NBA2U")
label=tk.Label(win, text="Please Choose a Picture",font=7,height= 10,width=50)   #Build a label item
label.grid(row = 1, column = 1)

options = {}
Opemfilename = []
i= 0
button=tk.Button(win,text="Load Picture",command=clickImport,font=7,height= 4,width=10)
button.grid(row = 2, column = 1)

image_frame = tk.Frame(win)
image_file = im = image_label = None
image_frame.grid(row = 4, column = 1)

buttonCNN=tk.Button(win,text="Predict (CNN)",command=CNN,font=7,height= 4,width=10)
buttonCNN.grid(row = 1, column = 2)

win.mainloop()



