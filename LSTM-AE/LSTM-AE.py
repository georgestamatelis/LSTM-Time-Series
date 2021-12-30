import numpy as np
import keras 
from keras.optimizers import adam_v2
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed,Dropout
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import initializers

from keras.utils.vis_utils import plot_model
train_path="../Data/nasd_input.csv"
test_path="../Data/nasd_query.csv"

train_file=open(train_path,"r")
allSeries=[]
for line in train_file:
    time_serie=line.split()
    time_serie.pop(0)
    time_serie=[float(x) for x in time_serie]
    #print("time_serie shape=",np.array(time_serie).shape,"vs",len(time_serie),"vs",len(line.split()))
    allSeries.append(np.array(time_serie))

#trainSet=np.vstack(allSeries)
n_in=730
X=np.vstack(allSeries)
scaler = StandardScaler().fit(X)
X=scaler.transform(X)
array_sum = np.sum(X)
array_has_nan = np. isnan(array_sum)
print("X has nan",array_has_nan)

print(X)
#X=np.random.rand(100,730)
print("X.shape=",X.shape)
# prepare output sequence
X = X.reshape((100, n_in, 1))
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
model.fit(X, X, epochs=100, verbose=1)
plot_model(model, show_shapes=True, to_file='LSTM-AE.png')
# demonstrate recreation
yhat = model.predict(X, verbose=0)
print(yhat[0,:,0])