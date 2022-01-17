
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

#read user arguments
train_path="../Data/nasdaq2007_17.csv" #default
num_series=4 #default
for i in range(len(sys.argv)):
  if sys.argv[i]=="-d":
    train_path=sys.argv[i+1]
  if sys.argv[i]=="-n":
    num_series=int(sys.argv[i+1])
    
  



#function that reads the dataset
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
data,n_in=read_file(train_path)
train_len=int(0.8*n_in)

test_len=n_in-train_len
train_set=data[:,:train_len]

test_set=data[:,-test_len:]



answer=input("Do you want to fit the NN on the entire train set? (y/n)")
if answer=="y":
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


  #load the model
  model=keras.models.load_model("large.h5")

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
  indexes=random.sample(range(data.shape[0]),num_series)

  for index in indexes:    
    display_pred(index)
    plt.show()
#one model for each time serie
elif answer=="n":
  models=[]
  scalers=[]
  look_back=10
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
  #train a model for each time serie
  models=[]
  for i  in range(20):
    name="model-"+str(i+1)+".h5"
    models.append(keras.models.load_model(name))
  
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
  indexes=random.sample(range(20),num_series)
  for index in indexes:
    plot_stock(index)
    plt.show()
  

else:
  print("wrong answer")


