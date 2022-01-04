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
THRESHOLD=0.0015
train_path="../Data/nasdaq2007_17.csv"
numSeries=320
for i in range(len(sys.argv)):
  if sys.argv[i]=="-mae":
    THRESHOLD=float(sys.argv[i+1])
  if sys.argv[i]=="-d":
    train_path=sys.argv[i+1]
  if sys.argv[i]=="-n":
    numSeries=int(sys.argv[i+1])
  



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
stock=test[0]
test=test.reshape(test.shape[0],n_in,1)
pred=model.predict(test)
print("TEST MAE:",np.absolute(np.subtract(test, pred)).mean())

#NOW TIME TO PLOT SOME ANOMALIES
def anomalies_in_stock(index):
  stock=test[index]
  stock=stock.reshape(1,n_in,1)
  originalStock=sc.inverse_transform(stock.reshape(1,n_in))

  predStock=model.predict(stock)
  anomaliesX=[]
  anomaliesY=[]
  for i in range(n_in):
    if np.absolute(stock[0][i][0]-predStock[0][i][0]) > THRESHOLD:
      anomaliesX.append(i)
      anomaliesY.append(originalStock[0][i])
  #anomalies=np.array(anomalies)
  plt.plot(np.linspace(0, n_in,n_in),originalStock[0])
  plt.scatter(anomaliesX,anomaliesY,color="r")  
  #plt.show()
from matplotlib.pyplot import figure

figure(figsize=(8, 6), dpi=80)

plt.subplot(2, 2, 1)
anomalies_in_stock(1)
plt.subplot(2,2,2)
anomalies_in_stock(2)
plt.subplot(2,2,3)
anomalies_in_stock(0)
plt.subplot(2,2,4)
anomalies_in_stock(7)
plt.show()