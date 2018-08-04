
# coding: utf-8

# In[ ]:


import numpy as np
import math
import os
import pandas as pd

import random
import tensorflow
import keras
from sklearn import metrics
#!pip install peakutils
#import peakutils
from sklearn.utils import shuffle
import tensorflow as tf
from scipy.interpolate import *
from scipy.signal import *
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GaussianDropout
from keras.layers import Conv2D
from keras.layers import Conv1D
from keras.layers import MaxPool1D
from keras.layers import Dropout
from keras.layers import MaxPool2D
from keras.optimizers import SGD


# In[ ]:


large_data = pd.DataFrame({"Signal":[], "Label":[]})
large_data

datafiles = ['s20011.xz','s20131.xz','s20251.xz','s20341.xz','s20471.xz','s20591.xz','s30721.xz','s20021.xz','s20141.xz',
's20351.xz','s20481.xz','s20601.xz','s30731.xz','s20031.xz','s20151.xz','s20271.xz','s20361.xz','s20491.xz','s20621.xz',
's20272.xz','s20371.xz','s20501.xz','s20431.xz','s20551.xz','s20181.xz','s30661.xz','s20301.xz','s20231.xz','s20321.xz',
's20631.xz','s30741.xz','s20051.xz','s20171.xz','s20273.xz','s20381.xz','s20511.xz','s20641.xz','s30742.xz','s20061.xz',
's20274.xz','s20391.xz','s20521.xz','s20651.xz','s30751.xz','s20071.xz','s20191.xz','s20281.xz','s20401.xz','s20531.xz',
's30752.xz','s20081.xz','s20201.xz','s20291.xz','s20411.xz','s20541.xz','s30671.xz','s30761.xz','s20091.xz','s20211.xz',
's30681.xz','s30771.xz','s20101.xz','s20221.xz','s20311.xz','s20441.xz','s20561.xz','s30691.xz','s30781.xz','s20111.xz',
's20451.xz','s20571.xz','s30701.xz','s30791.xz','s20121.xz','s20241.xz','s20331.xz','s20461.xz','s20581.xz','s30711.xz', 
's30732.xz','s20041.xz','s20161.xz','s30801.xz', 's20261.xz',]

os.chdir('../processed_data')
for i in datafiles:
	dat = pd.read_pickle(i)
	for count, signal in enumerate(dat['Signal']):
		dat['Signal'][count] = dat['Signal'][count] - np.mean(dat['Signal'][count])
		dat['Signal'][count] = dat['Signal'][count] / np.std(dat['Signal'][count])
	print(i)
	large_data = large_data.append(dat)
os.chdir('../')


# In[ ]:


large_data = large_data[large_data.Label != 'scct'] #dropping rows with shifts, we are not classifying
large_data = large_data[large_data.Label != 'sst']

large_data['Label'] = large_data['Label'].map({'st': 0, 'rtst': 1, 'normal': 2})

print("# of data rows for ST: ", len(large_data.loc[large_data['Label'] == 0]))
print("# of data rows for RTST: ", len(large_data.loc[large_data['Label'] == 1]))
print("# of data rows for Normal: ", len(large_data.loc[large_data['Label'] == 2]))

newx = 0
for i in large_data['Signal']:
	if len(i)!= 250:
		print(len(i))
		newx+=1
print("nx", newx)

#convert to ints

large_data['Label'] = large_data['Label'].astype('category').cat.codes

#randomly shuffle dataframe

large_data = large_data.sample(frac=1).reset_index(drop=True)


# In[ ]:


y = large_data['Label'].values
X = []
for i in large_data['Signal']:
    X.append(i)
X = np.array(X)

print("Y")
print(y.shape)
print("X")
print(X.shape)


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)

y_train = keras.utils.to_categorical(y_train, 3)
y_val = keras.utils.to_categorical(y_val, 3)


y_train = y_train.astype('float64')#.reshape((y_train.shape[0], y_train.shape[1], 1))
y_val = y_val.astype('float64')#.reshape((y_val.shape[0], y_val.shape[1], 1))
X_train = X_train.astype('float64').reshape((X_train.shape[0], X_val.shape[1], 1))
X_val = X_val.astype('float64').reshape((X_val.shape[0], X_val.shape[1], 1))

print("Training")
print(X_train.shape, y_train.shape)
print("Validation")
print(X_val.shape, y_val.shape)


# In[ ]:


print(X_train[0].shape)

model = keras.models.Sequential()
model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=X_train[0].shape))
model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=3, padding='valid'))
model.add(GaussianDropout(.25))
model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(GaussianDropout(.25))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()


# In[ ]:


print(y_train.shape)


# In[ ]:


print(y_train.shape)
model.fit(X_train, y_train, epochs= 12, batch_size= 2, validation_data=(X_val, y_val), verbose=1)


# In[ ]:


#Save the model

model.save("BWSI2018model_v3_1.h5")

model.save_weights('BWSI2018model_v3_1_weights.h5')

with open('BWSI2018model_architecture.json', 'w') as f:
    f.write(model.to_json())

