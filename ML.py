import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense
import random
"""
data = pd.read_csv("C:/Users/Workstation/git/ML/data/acc-data.csv")
data
data.columns
"""
c=np.array([-40,-10,0,8,15,22,38,])
f=np.array([-40,14,32,46,59,72,100,])
model = keras.Sequential()
model.add(Dense(units=1, input_shape=(1,),activation='linear'))

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.1))
history = model.fit(c,f,epochs=500, verbose=0 )

plt.plot(history.history['loss'])
plt.grid(1)
plt.show()
print(model.predict([100]))
print(model.get_weights())