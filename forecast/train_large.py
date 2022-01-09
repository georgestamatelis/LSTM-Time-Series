
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
print("Data.shape=",data.shape)
train_len=int(0.8*n_in)

test_len=n_in-train_len
train_set=data[:,:train_len]

test_set=data[:,-test_len:]


#early stopping 
#https://stackoverflow.com/questions/37293642/how-to-tell-keras-stop-training-based-on-loss-value
class EarlyStoppingByLossVal(Callback):
  def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
    super(Callback, self).__init__()
    self.monitor = monitor
    self.value = value
    self.verbose = verbose

  def on_epoch_end(self, epoch, logs={}):
    current = logs.get(self.monitor)
    if current is None:
      warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

    if current < self.value:
      if self.verbose > 0:
        print("Epoch %05d: early stopping THR" % epoch)
      self.model.stop_training = True
callbacks = [
  EarlyStoppingByLossVal(monitor='loss', value=0.25, verbose=1),
  ]
training_set_scaled=[]
test_set_scaled=[]
#scale each time serie
scalers=[]
for i in range(0,data.shape[0]):
    train=train_set[i].reshape(train_set[i].shape[0],1)
    test=test_set[i].reshape(test_set[i].shape[0],1)
    sc = MinMaxScaler(feature_range = (0, 1)).fit(train)
    scalers.append(sc)
    train=sc.transform(train)
    test=sc.transform(test)
    training_set_scaled.append(train)
    test_set_scaled.append(test)
training_set_scaled=np.array(training_set_scaled)
test_set_scaled=np.array(test_set_scaled)

#create the training dataset using look back
look_back=10
xTrain = []
yTrain = []
for row in training_set_scaled:
    for i in range(look_back, train_len):
        xTrain.append(row[i-look_back:i])
        yTrain.append(row[i])
xTrain, yTrain = np.array(xTrain), np.array(yTrain)


#define the model
n_in=xTrain.shape[1] #number of features

xTrain = xTrain.reshape((xTrain.shape[0], n_in, 1))
model = Sequential()
model.add(LSTM(units=25, activation='relu', input_shape=(n_in,1)))
model.add(Dropout(0.2))
model.add(Dense(1))
#define the optimizer
opt = adam_v2.Adam(learning_rate=0.01)

model.compile(optimizer=opt, loss='mse')


  
#train the model
model.fit(xTrain,yTrain,epochs=200,verbose=1,callbacks=callbacks)
model.save("large.h5")
#choose some random stocks from the test set and predict them


def display_pred(index):
    stock=test_set_scaled[index]
    xTest=[]
    for i in range(look_back,test_len):
      xTest.append(stock[i-look_back:i,0])
    xTest=np.array(xTest)
    xTest=xTest.reshape(xTest.shape[0],xTest.shape[1],1)
    pred=model.predict(xTest)

    pred=scalers[index].inverse_transform(pred)

    initial_stock=test_set[index][look_back:]
    plt.plot(np.linspace(0, test_len-look_back,test_len-look_back),initial_stock)
    plt.plot(np.linspace(0, test_len-look_back,test_len-look_back),pred)


indexes=random.sample(range(data.shape[0]),3)

for index in indexes:    
    display_pred(index)
    plt.show()