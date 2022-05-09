'''
HOG & SVM
'''

# import the necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_ubyte, img_as_float,data, exposure
from skimage.feature import hog
import tensorflow as tf
import cv2
from keras.preprocessing import image
import pandas as pd
from tqdm import tqdm
from sklearn import svm, metrics
from sklearn.utils import shuffle



from base import LoadDataHOG

X_train,X_train_HOG,X_train_HOG_flattened,y_train,X_test,X_test_HOG,X_test_HOG_flattened,y_test =LoadDataHOG(100)
# X_test_flattened = X_test.reshape(len(X_train),-1)

print("X_train ",X_train.shape)
print("X_train_HOG ",X_train_HOG.shape)
print("X_train_HOG_flattened ",X_train_HOG_flattened.shape)
print("y_train ", y_train.shape)
print("---------------------------")
print("X_test ",X_test.shape)
print("X_test_HOG ",X_test_HOG.shape)
print("X_test_HOG_flattened ",X_test_HOG_flattened.shape)
print("y_test ", y_test.shape)

'''
create SVM classifier
'''

# from train_SVM import train_linear_SVM
# Create a classifier: a support vector classifier
# classifier = svm.SVC(gamma=0.001)
classifier = svm.SVC(kernel='linear')

print('X_train shape =', X_train.shape)
print('X_train_HOG shape =', X_train_HOG.shape)
print('y_train shape =', y_train.shape)
# X_train shape = (20000, 784)
# y_train shape = (20000,)

X_train_flattened = X_train.reshape(len(X_train),-1)
X_train_HOG_flattened = X_train_HOG.reshape(len(X_train),-1)
print('X_flatenned =', X_train_flattened.shape)
print('X_HOG_flatenned =', X_train_HOG_flattened.shape)

# Train the classifier
classifier.fit( X_train_HOG_flattened ,y_train)


'''
Preict using SVM Classifier and test data
'''
# predict the classes on the images of the test set
y_pred = classifier.predict(X_test_HOG_flattened)

X_test, y_test, y_pred = shuffle(X_test, y_test, y_pred)
X_test_img = X_test

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6), sharex=True, sharey=True)
ax = axes.ravel()

for i in range(10):
    ax[i].imshow(X_test_img[i], cmap='gray')
    ax[i].set_title(f'Label: {y_test[i]}\n Prediction: {y_pred[i]}')
    ax[i].set_axis_off()
fig.tight_layout
plt.show()

print(f"""Classification report for classifier {classifier}:\n
      {metrics.classification_report(y_test, y_pred)}""")