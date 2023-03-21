import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Reshape, Conv1D, LSTM, BatchNormalization, Dropout, GRU
from keras.models import Sequential
from tensorflow.random import set_seed
from tensorflow.keras.callbacks import ReduceLROnPlateau
import random
data = pd.read_csv("C:/Users/Workstation/git/ML/data/acc-data.csv")
data.columns
len(data[data['Classs']==1])/50
n = 39
X = np.empty((n,50,3))
Y = np.empty(n)
for i in range(0,n):
    X[i]=data.iloc[i*50:(i+1)*50,2:5]
    Y[i]=data.iloc[i*50,5]
print("Training data input:")
print(X)
print("Training data results:")
print(Y)
from sklearn.model_selection import train_test_split
X_train, X_test,Y_train, Y_test = train_test_split(X,Y, test_size=0.3,random_state=8, shuffle=True)
X_train.shape

def get_model():
    set_seed(2)
    model = Sequential([
        Dense(50, activation='relu',input_shape=(50,3)),        
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy',
                optimizer=Adam(learning_rate=0.01),metrics=['accuracy',keras.metrics.Precision(),keras.metrics.Recall()])
    return model
k=3
size = X_train.shape[0] // k
avg_scores = []
for i in range(k):
    
    print("fold : ", i)
    model = get_model()
    
    X_train_ = np.concatenate([X_train[:i*size] , X_train[(i+1)*size:]],axis=0)
    X_val = X_train[i*size: (i+1)*size]
    
    Y_train_ = np.concatenate([Y_train[:i*size] , Y_train[(i+1)*size:]])
    Y_val = Y_train[i*size: (i+1)*size]
    
    model = get_model()
    
    model.fit(X_train_,Y_train_,epochs=64)
    scores = model.evaluate(X_val, Y_val)
    avg_scores.append(scores)

np.average(avg_scores,axis=0)
model = get_model()
model.fit(X_train,Y_train,epochs=64)
score=model.evaluate(X_test,Y_test)
print(model.metrics_names[1])
score[1]
model.save('AI.model')