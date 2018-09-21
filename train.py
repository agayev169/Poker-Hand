# val_acc = 0.9945 with len(test_data) = 25010 and len(val_data) = 100001

import csv

def prepare_data(data):
	X, y = [], []
	try:
		for i in range(10):
			X.append(float(data[i]))
		y.append(float(data[10]))
	except:
		pass
	return X, y


file = open('data_train.csv', 'r')
r = csv.reader(file)

X, y = [], []

for row in r:
	X_r, y_r = prepare_data(row)
	if len(X_r) == 10 and len(y_r) == 1:
		X.append(X_r)
		y.append(y_r)

# print(len(X))
# print(len(y))


file = open('data_val.csv', 'r')
r = csv.reader(file)

X_val, y_val = [], []

count = 0

for row in r:
	if count > 100000:
		break
	X_r, y_r = prepare_data(row)
	if len(X_r) == 10 and len(y_r) == 1:
		X_val.append(X_r)
		y_val.append(y_r)
	count += 1


import numpy as np

X = np.array(X)
y = np.array(y)
X_val = np.array(X_val)
y_val = np.array(y_val)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import normalize, to_categorical
from tensorflow.keras.optimizers import Adam

X = normalize(X)
X_val = normalize(X_val)

y = normalize(y)
y_val = normalize(y_val)

# print(y_val.shape)
# print(y.shape)

model = Sequential()

model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(y.shape[1], activation = 'sigmoid'))

opt = Adam(lr = 0.001)
model.compile(optimizer = opt, loss = 'mean_squared_error', metrics = ['accuracy'])
model.fit(X, y, batch_size = 64, epochs = 200, validation_data = (X_val, y_val))
