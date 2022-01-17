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

import random

import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

import sys

#reproducability

from numpy.random import seed
seed(185)
import tensorflow
tensorflow.random.set_seed(185)

#read command line argumens
THRESHOLD=0.0015 #default value
train_path="../Data/nasdaq2007_17.csv"
numSeries=5
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
#load  data
data=read_file(train_path)
#scale data
sc = MinMaxScaler(feature_range = (0, 1)).fit(data)
data=sc.transform(data)

#load the model
model=keras.models.load_model("LSTM-AE")

"""
    TIME TO PLOT THE ANOMALIES
"""
n_in=data.shape[1]
def anomalies_in_stock(index):
  stock=data[index]
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


indexes=random.sample(range(data.shape[0]),numSeries)
for index in indexes:
    anomalies_in_stock(index)
    plt.show()
