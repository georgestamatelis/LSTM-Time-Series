import sys, os
from sys import argv
from csv import reader

import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import models, layers,optimizers, losses, metrics
from keras.models import Model,load_model
from keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.optimizers import adam_v2

#reproducability
from numpy.random import seed
seed(185)
import tensorflow
tensorflow.random.set_seed(185)

#utility functions
def read_params():
  print("Please provide the hyperparameters:")
  nodes = int(input("Number of nodes for the fully connected layer: "))
  epochs = int(input("Number of epochs: "))
  batch_size = int(input("Batch size: "))
  return nodes,epochs,batch_size

def read_data(dataset_path):
    dataset=open(dataset_path,"r")
    train_data = []
    with open(dataset_path, 'r') as read_obj:
        for row in dataset:
            serie=row.split()
            serie.pop(0) #throw way the time serie name
            serie=[float(x) for x in serie]
            train_data.append(np.array(serie))
    train_data=np.vstack(train_data)
    return train_data

def scale_data(train,test):
    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    train=scaler.fit_transform(train)
    test =scaler.transform(test)
    return train,test

def preprocces_data(data):
    x=data.shape[0]
    y=data.shape[1]
    X = data.reshape((x,y, 1))
    return X,y

#defining autoencoder layers
def encoder(input_data):
    conv1 = Conv1D(3650, 10, activation='relu', padding='same')(input_data)
    conv1 = BatchNormalization()(conv1)

    conv1 = Conv1D(1825, 10, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)

    conv1 = MaxPooling1D(5)(conv1)
    conv1 = Dropout(0.2)(conv1)

    conv1 = Conv1D(912, 10, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
  
    conv2 = Conv1D(456, 10, activation='relu', padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv1D(1, 10, activation='relu', padding='same',name='bottleneck_layer')(conv2)
  
    return conv2

def decoder(encoded):
    conv = Conv1D(456, 10, activation='relu', padding='same')(encoded)
    conv = BatchNormalization()(conv)
    
    conv = Conv1D(912, 10, activation='relu', padding='same')(conv)
    conv = BatchNormalization()(conv)
  
    conv = UpSampling1D(5)(conv)
    conv = Dropout(0.2)(conv)

    conv = Conv1D(1825, 10, activation='relu', padding='same')(conv)
    conv = BatchNormalization()(conv)

    conv = Conv1D(3650, 10, activation='relu', padding='same')(conv)
    conv = BatchNormalization()(conv)
  
    decoded = Conv1D(1, 10, activation='sigmoid', padding='same')(conv)
    return decoded
#parse comand line arguments
dataset_path= ''
for index, argument in enumerate(argv):
    if argument == '-d': dataset_path = argv[index+1]

data = read_data(dataset_path) #read file
#split data to train and validation set
train=data[:300,:]
test=data[300:,:]
train,test = scale_data(train,test) #scale file data
X,y=preprocces_data(train)#reshape train data 
X_test,y_test = preprocces_data(test)#reshape validation data

#define parameters and model
input_data = Input(shape=(y, 1))
autoencoder = keras.Model(input_data, decoder(encoder(input_data)))
opt = adam_v2.Adam(learning_rate=0.00001)
# opt = RMSprop(learning_rate=0.000001)
autoencoder.compile(loss='mean_squared_error', optimizer = opt )

# Train the model
autoencoder_train = autoencoder.fit(X,X, batch_size=10,epochs=4,verbose=1,validation_data=(X_test,X_test),validation_batch_size=10)

#Save the model
autoencoder.save('model.h5')
autoencoder.summary()

#Plot the loss
plt.plot(autoencoder_train.history['loss'])
plt.plot(autoencoder_train.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim(0,1)
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

