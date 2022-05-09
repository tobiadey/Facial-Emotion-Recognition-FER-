'''
CNN
https://github.com/tobiadey/AICoursework/blob/main/app/main/cnn/run.py
'''

# import the necessary libraries
from gc import callbacks
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import tensorflow as tf
import cv2
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout,BatchNormalization,Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

from base import LoadData

X_train,X_train_flattened,y_train,X_test, X_test_flattened,y_test = LoadData(10000)

#function to the train the datasets, sample(used for testing purposes)  and actual
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = 0.2, random_state = 1)

print("X_train ",X_train.shape)
print("X_train_flattened ",X_train_flattened.shape)
print("y_train ", y_train.shape)
print("---------------------------")
print("X_test ",X_test.shape)
print("X_test_flattened ",X_test_flattened.shape)
print("y_test ", y_test.shape)

'''
create CNN classifier
'''

classifier = Sequential()

# classifier.add(Conv2D(filters=64, kernel_size=3, padding = 'same', activation='relu'))
# classifier.add(MaxPooling2D(pool_size=2, strides=2))
# classifier.add(Flatten())
# classifier.add(Dense(units=128, activation='relu'))
# classifier.add(Dense(8, activation='softmax'))


classifier.add(Conv2D(32,(3,3), activation='relu', input_shape=(100, 100, 3)))
classifier.add(MaxPooling2D(2,2))
# classifier.add(BatchNormalization())

classifier.add(Conv2D(64,(3,3), activation='relu', input_shape=(100,100,3)))
classifier.add(MaxPooling2D(2,2))
# classifier.add(BatchNormalization())

classifier.add(Conv2D(128,(3,3), activation='relu', input_shape=(100,100,3)))
classifier.add(MaxPooling2D(2,2))
# classifier.add(BatchNormalization())

classifier.add(Flatten())
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(8, activation='softmax'))
# 8 means labels range from 0-8

# # Compile the model
classifier.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])# metrics = What you want to maximise

history= classifier.fit(X_train, y_train,epochs=40,validation_data=(X_validate, y_validate))


# '''
# Preict using CNN Classifier and test data
# '''
'''
validation
'''
print("\n-----------------------------CNN model accuracy (Validation test) ----------------------------------")
score = classifier.evaluate(X_validate, y_validate, verbose=0)

print('validation loss: {:.2f}'.format(score[0]))
print('validation acc: {:.2f}'.format(score[1]))

# Plot training & validation loss values

plt.plot(history.history['loss'])
plt.title('Model loss/accuracy for Validation data')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Loss'], loc='upper left')

plt2 = plt.twinx()
color = 'red'
plt2.plot(history.history['accuracy'], color=color)
plt.ylabel('Accuracy')
plt2.legend(['Accuracy'], loc='upper center')
plt.show()


'''
testing
'''
print("\n-----------------------------CNN model accuracy (Testing data) ----------------------------------")
# predict the classes on the images of the test set
# y_pred = classifier.predict(X_test_HOG_flattened)
score = classifier.evaluate(X_test, y_test, verbose=0)
print('test loss: {:.2f}'.format(score[0]))
print('test acc: {:.2f}'.format(score[1]))
print('score: ',score)

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss/accuracy for Testing data')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Loss'], loc='upper left')
plt2 = plt.twinx()
color = 'red'
plt2.plot(history.history['accuracy'], color=color)
plt.ylabel('Accuracy')
plt2.legend(['Accuracy'], loc='upper center')
# plot accuracy

plt.show()

plt.subplot(212)
plt.title('Classification Accuracy')
plt.plot(history.history['accuracy'], color='blue', label='train')
plt.plot(history.history['val_accuracy'], color='orange', label='test')


plt.show()

# sift des, descrirbes the kp 
# grrey scale simplifies this, much more faster to use greyscale


# svm
# svm overfit easiely 
# svm multiclaass is bad changes it to 1 and 0 cnn better can handle multiclass 
# svm not goof for multiclass classification

# deep learning
# use validation as this is good to monitor for overfitting or not

# deep leanring is better the more layers you have the betterrr accuracy until it starts overfitting at some point
# as layers seperate the data and then do so much it can overfit
# just need to play around to find the best fit

# sklearn spilt into validate and all
# data loader is good
# try to seed to make sure to get same value on split


# mlp
# mlp can overfit easily over cnn as more training params

# cnn
# good for multiclass classification


# loss is how your prediction is different from the original value
# regularization can be used if data is overfitting
# take training set 20 80, keep chaining the ..cross validation


# # https://stackoverflow.com/posts/56909577/revisions
# # https://stackoverflow.com/questions/44151760/received-a-label-value-of-1-which-is-outside-the-valid-range-of-0-1-python/56909577#56909577
# classifier.add(Dense(8, activation='softmax'))
# # 8 means labels range from 0-8

# # # Compile the model
# classifier.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])# metrics = What you want to maximise
