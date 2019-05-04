import pandas as pd
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import GRU
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import math
import keras
from keras.layers import Dense, Activation
    
def rnn(x,y,h,t):
    def modify(dataset,d, step_size):
        data_X, data_Y = [], []
        for i in range(len(dataset)-step_size-1):
            a = dataset[i:(i+step_size), :]
            data_X.append(a)
            data_Y.append(d[i + step_size, :])
        return np.array(data_X), np.array(data_Y)

    x_scale,y_scale = MinMaxScaler()
    X = x_scale.fit_transform(x)
    y=y.reshape(y.shape[0],1)
    Y = y_scale.fit_transform(y)
    Y=Y.reshape(756,1)
    X=X.reshape(756,2)
    split=int(0.8*len(X))
    X_train=X[:split,:]
    X_test=X[split:,:]
    y_train=Y[:split,:]
    y_test=Y[split:,:]
    X_train=X_train.reshape(X_train.shape[0],2)
    X_test=X_test.reshape(X_test.shape[0],2)
    trainX, trainY = modify(X_train,y_train, t)
    testX, testY = modify(X_test,y_test, t)
    trainX = np.reshape(trainX, (trainX.shape[0],trainX.shape[1],2))
    testX = np.reshape(testX, (testX.shape[0],testX.shape[1],2))
    model = Sequential()
    model.add(keras.layers.LSTM(h, input_shape=(t,2), return_sequences = True))
    model.add(keras.layers.LSTM(h))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam') 
    model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)
    testP = model.predict(testX)
    testP = y_scale.inverse_transform(testP)
    testY = y_scale.inverse_transform(testY)
    plt.plot(range(1,testY.shape[0]+1), testY,label='actual')
    plt.plot(range(1,testY.shape[0]+1), testP[:,0],label='prediction')
    plt.legend()
    plt.show()
    
    
