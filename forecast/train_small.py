#NECESSARY IMPORTS
from keras.callbacks import Callback
import numpy as np
import keras 
from keras.optimizers import adam_v2
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed,Dropout
from numpy import testing
from numpy.core.numerictypes import sctype2char
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import random
from keras.optimizers import adam_v2
import sys


from tensorflow.keras import initializers


#REPRODUCABILITY

from numpy.random import seed
seed(185)
import tensorflow
tensorflow.random.set_seed(185)

#a function that reads the dataset
def read_file(filePath):
    fl=open(filePath,"r")
    allSeries=[]
    for line in fl:
        time_serie=line.split()
        time_serie.pop(0)
        time_serie=[float(x) for x in time_serie]
        allSeries.append(np.array(time_serie))

    #trainSet=np.vstack(allSeries)
    n_in=allSeries[0].shape[0]
    data=np.vstack(allSeries)
    return data,n_in 
#load and split the data
train_path="../Data/nasdaq2007_17.csv" #default

data,n_in=read_file(train_path)


train_len=int(0.8*n_in)

test_len=n_in-train_len
train_set=data[:,:train_len]

test_set=data[:,-test_len:]



models=[]
scalers=[]
look_back=10
training_set_scaled=[]
test_set_scaled=[]
#scale each time serie
scalers=[]

#because we can not store 365 models we only train on the first 20 series
for i in range(0,20):
    train=train_set[i].reshape(train_set[i].shape[0],1)
    test=test_set[i].reshape(test_set[i].shape[0],1)
    sc = MinMaxScaler(feature_range = (0, 1)).fit(train)
    scalers.append(sc)
    train=sc.transform(train)
    test=sc.transform(test)
    training_set_scaled.append(train)
    test_set_scaled.append(test)
#train a model for each time serie
models=[]
for i  in range(20):
    train=training_set_scaled[i]
    test=test_set_scaled[i]
    xTrain=[]
    yTrain=[]
    for j in range(look_back,train.shape[0]):
      xTrain.append(train[j-look_back:j])
      yTrain.append(train[j])
    xTrain,yTrain=np.array(xTrain),np.array(yTrain)
    print(xTrain.shape,yTrain.shape)

    xTrain = xTrain.reshape((xTrain.shape[0], xTrain.shape[1], 1))
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(xTrain.shape[1],1)))
    #model.add(Dropout(0.2))
    model.add(Dense(1))
    opt = adam_v2.Adam(learning_rate=0.01)

    model.compile(optimizer=opt, loss='mse')

    model.fit(xTrain,yTrain,epochs=1,verbose=1)
    models.append(model)
#some plots to make sure training went ok 
def plot_stock(index):
    stock=test_set_scaled[index]
    xTest=[]
    yTest=[]
    for i in range(look_back,stock.shape[0]):
      xTest.append(stock[i-look_back:i])
      yTest.append(stock[i])
    xTest,yTest=np.array(xTest),np.array(yTest)
    pred=models[index].predict(xTest)
    pred=scalers[index].inverse_transform(pred)

    initial_stock=test_set[index][look_back:]
    plt.plot(np.linspace(0, test_len-look_back,test_len-look_back),initial_stock)
    plt.plot(np.linspace(0, test_len-look_back,test_len-look_back),pred)

indexes=random.sample(range(20),3)
for index in indexes:
    plot_stock(index)
    plt.show()
#now save the models
i=1
for model in models:
    name="model-"+str(i)+".h5"
    i=i+1
    model.save(name)