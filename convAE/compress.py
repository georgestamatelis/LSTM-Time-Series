import sys, os
from sys import argv
from csv import reader

import keras
import tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import models, layers,optimizers, losses, metrics
from keras.models import Model,load_model
from keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


#utility functions
def read_data(dataset_path):
    dataset=open(dataset_path,"r")
    train_data = []
    names = []
    with open(dataset_path, 'r') as read_obj:
        for row in dataset:
            serie=row.split()
            a =serie.pop(0)
            names.append(a) #save the names of the series 
            serie=[float(x) for x in serie]
            train_data.append(np.array(serie))
    return train_data, names

def scale_data(X):
    X=np.vstack(X)
    scaler = MinMaxScaler()
    X=scaler.fit_transform(X)
    return X

def preprocces_data(data):
    x=data.shape[0]
    y=data.shape[1]
    X = data.reshape((x,y, 1))
    return X,y
#parse comand line arguments
dataset_path= ''
for index, argument in enumerate(argv):
    if argument == '-d': dataset_path = argv[index+1]
    if argument == '-od': input_out_path = argv[index+1]
    if argument == '-oq': query_out_path = argv[index+1]

if not dataset_path: dataset_path =input('Please provide the file path of the input/query data: ')
if not input_out_path: input_out_path = input('Please provide the path to store the encoded input data: ')
if not query_out_path: query_out_path = input('Please provide the path to store the encoded query data: ')
model_path = input('Please provide the path of the training model: ')

#read and prepare data
data,names = read_data(dataset_path) #read file
data=scale_data(data)#scale data
X,y = preprocces_data(data)#reshape data

#load encoder
AE = load_model(model_path)
encoder_output = AE.get_layer('bottleneck_layer').output

#build a model
encoder_model = keras.Model(AE.input,encoder_output)
#encode all data
encoded_data = encoder_model.predict(X) 
encoded_data = encoded_data.reshape(encoded_data.shape[0],encoded_data.shape[1])

#split on input and query and save to files
final_data = np.column_stack((names, encoded_data))
input,query = np.split(final_data,[349])

np.savetxt("encoded_input.csv", input ,fmt='%s', delimiter="\t")
np.savetxt("encoded_query.csv", query ,fmt='%s', delimiter="\t")

encoder_model.summary()