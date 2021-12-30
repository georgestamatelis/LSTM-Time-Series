from os import XATTR_CREATE
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


from tensorflow.keras import initializers

from keras.utils.vis_utils import plot_model

#reproducability

from numpy.random import seed
seed(185)
import tensorflow
tensorflow.random.set_seed(185)
##read model

train_path="../Data/nasd_input.csv"
test_path="../Data/nasd_query.csv"

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
    n_in=allSeries[0].shape
    print("n_in=",n_in)
    data=np.vstack(allSeries)
    return data 
#load and scale training data
data=read_file(train_path)
scaler = StandardScaler().fit(data)
data=scaler.transform(data)
array_sum = np.sum(data)
array_has_nan = np. isnan(array_sum)
if array_has_nan==True:
    print("ERROR IN DATASET")
    exit()

#split data and label
#goal is to use first prices to predict last price
look_back=50
xTrain=[]
yTrain=[]
total_length=data.shape[1]

for stock in data:
    #print("stock.shape=",stock.shape)
    for i in range(look_back,total_length-look_back):
        xTrain.append(stock[i-look_back:i])
        yTrain.append(stock[i])

xTrain=np.array(xTrain)
yTrain=np.array(yTrain)
print("X.shape=",xTrain.shape,"y.shape=",yTrain.shape)

#define the model
n_in=xTrain.shape[1] #number of features

xTrain = xTrain.reshape((xTrain.shape[0], n_in, 1))
model = Sequential()
model.add(LSTM(units=25, activation='relu', input_shape=(n_in,1)))
model.add(Dropout(0.2))
model.add(Dense(1))
#define the optimizer
from keras.optimizers import adam_v2
opt = adam_v2.Adam(learning_rate=0.01)

model.compile(optimizer=opt, loss='mse')


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
    # EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    #ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
]
model.fit(xTrain,yTrain,epochs=200,verbose=1,callbacks=callbacks)

#read test data and scale them
testData=read_file(test_path)
testData=scaler.transform(testData)
test_data_len=testData.shape[1]
xTest=[]
yTest=[]
for stock in testData:
    #print("stock.shape=",stock.shape)
    for i in range(look_back,test_data_len-look_back):
        xTest.append(stock[i-look_back:i])
        yTest.append(stock[i])

xTest=np.array(xTest)
yTest=np.array(yTest)

xTest = xTest.reshape((xTest.shape[0], n_in, 1))
print("Xtest shape=",xTest.shape)
yPred=model.predict(xTest)
#yPred=scaler.inverse_transform(yPred)

print("Test MSE:",mean_squared_error(yTest,yPred))

yPred=model.predict(xTrain)
print("Train MSE:",mean_squared_error(yTrain,yPred))
print("ypred.shape=",yPred.shape,"xTest.shape=",xTest.shape)
print("TIME TO PLOT")
#now some plots
original_test=scaler.inverse_transform(yTest)
predicted_stock_price = scaler.inverse_transform(yPred)
