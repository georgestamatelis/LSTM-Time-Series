

#necessary imports
import numpy as np
import keras 
from keras.optimizers import adam_v2
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed,Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import initializers

import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

import sys

#reproducability

from numpy.random import seed
seed(185)
import tensorflow
tensorflow.random.set_seed(185)

#read command line argumens
train_path="../Data/nasdaq2007_17.csv"
numSeries=340

  



def read_file(filePath):
    fl=open(filePath,"r")
    allSeries=[]
    for line in fl:
        time_serie=line.split()
        time_serie.pop(0)
        time_serie=[float(x) for x in time_serie]
        #print("time_serie shape=",np.array(time_serie).shape,"vs",len(time_serie),"vs",len(line.split()))
        allSeries.append(np.array(time_serie))

    #trainSet=np.vstack(allSeries)
    data=np.vstack(allSeries)
    return data 
#load and scale training data
data=read_file(train_path)
train=data[:numSeries,:]
test=data[numSeries:,:]
#scale data
sc = MinMaxScaler(feature_range = (0, 1)).fit(train)
train=sc.transform(train)
test=sc.transform(test)

n_in=train.shape[1]
# prepare output sequence
X = train.reshape((train.shape[0], n_in, 1))
# DEFINE THE MODEL
model = Sequential()
model.add(LSTM(units=250, activation='relu', input_shape=(n_in,1)))
model.add(Dropout(0.2))
model.add(RepeatVector(n_in))
model.add(LSTM(units=250, activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(units=1)))
from keras.optimizers import adam_v2
opt = adam_v2.Adam(learning_rate=0.00001)


model.compile(optimizer=opt, loss='mae')
# fit model
model.fit(X, X, epochs=2, verbose=1)


#print mae in test set to see if the model generalises well
test=test.reshape(test.shape[0],n_in,1)
pred=model.predict(test)
print("TEST MAE:",np.absolute(np.subtract(test, pred)).mean())

#save the model
model.save("LSTM-AE")
