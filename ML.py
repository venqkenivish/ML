import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense,Flatten
import random
data = pd.read_csv("C:/Users/Workstation/git/ML/data/acc-data.csv")
n = 39
X = np.empty((n,50*3))
Y = np.empty(n)
for i in range(0,n):
    for j in range(0,3):
        X[i,j*50:(j+1)*50]=data.iloc[i*50:(i+1)*50,j+2]
    Y[i]=data.iloc[i*50,5]
    
from sklearn.model_selection import train_test_split
X_train, X_test,Y_train, Y_test = train_test_split(X,Y, test_size=0.3,random_state=8, shuffle=True)
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
lr = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

print(lr.summary())   
lr.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])


lr.fit(X_train, Y_train, batch_size=32, epochs=5, validation_split=0.2)

lr.evaluate(X_test,Y_test)
pred_y = lr.predict(X_test)
print(accuracy_score(Y_test,pred_y))
print(precision_score(Y_test,pred_y))
print(recall_score(Y_test,pred_y))