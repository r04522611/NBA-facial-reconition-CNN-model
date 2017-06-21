
'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

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
import os
NBA = {'0':'JR Smith', '1':'Anthony Davis','2':'Jeremy Lin','3':'Kawhi Leonard',
       '4':'Kevin Love','5':'Kobe Bryant','6':'LeBron James','7':'Russel WestBrook',
       '8':'Stephen Curry','9':'Tony Parker'}
keras.backend = 'theano'

#set the batch_size num_classes epochs 
batch_size = 128
num_classes = 10
epochs = 20 

#read from the Test.csv for X&Y_test, X_predict Input 
MyTEST = np.genfromtxt('Test.csv',delimiter = ',',skip_header = 1)
X_test = MyTEST [:50,5:12]
Y_test = MyTEST [:50,12]
X_test = X_test.astype('float32')
Y_test = Y_test.astype('int')
YY_test= np_utils.to_categorical(Y_test,10) # turn the category (0-9) into 10bit     
X_predict =  MyTEST [50:,5:12]
X_predict =X_predict.astype('float32')

#print (len(X_predict))
#print (len(X_test))



#read from Training Data NBA3.csv

MyDATA = np.genfromtxt('Train.csv',delimiter = ',',skip_header = 1)
X = MyDATA[:,5:12]
Y= MyDATA[:,12]
                        # 確保資料型態正確
X = X.astype('float32')
Y = Y.astype('int')
#print(X[0,:])
#print(Y[0])
YY = np_utils.to_categorical(Y,10)
#print(YY[0,:])



# Build a sequential model for CNN training process

model = Sequential()
model.add(Dense(128, input_dim=7, activation='sigmoid'))
model.add(Dense(256,activation='sigmoid'))
model.add(Dense(10,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
history = model.fit(X, YY,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)


# use the Sequential model's function. evaluate to compute the accuracy

score = model.evaluate(X_test, YY_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Import the data for model prediction

Predict= model.predict(X_predict,batch_size=128)

#print (Predict)

#print out the prediction of outcome (should be 10x1 array of the probability)
outcome = np.argmax(Predict, axis=1)  #Max is the most likely feature
for i in outcome:
    print (NBA[str(i)])

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
