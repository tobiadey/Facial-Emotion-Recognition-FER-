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



from base import LoadDataSIFT

X_train,X_train_SIFT,X_train_SIFT_flattened,y_train,X_test,X_test_SIFT,X_test_SIFT_flattened,y_test =LoadDataSIFT(100)
# X_test_flattened = X_test.reshape(len(X_train),-1)

print("X_train ",X_train.shape)
print("X_train_SIFT ",X_train_SIFT.shape)
print("X_train_SIFT_flattened ",X_train_SIFT_flattened.shape)
print("y_train ", y_train.shape)
print("---------------------------")
print("X_test ",X_test.shape)
print("X_test_SIFT ",X_test_SIFT.shape)
print("X_test_SIFT_flattened ",X_test_SIFT_flattened.shape)
print("y_test ", y_test.shape)
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5), sharex=True, sharey=True)
ax = axes.ravel()

# for i in range(10):
#     ax[i].imshow(X_train_SIFT[i], cmap='gray')
#     ax[i].set_title(f'Label: {y_train[i]}')
#     ax[i].set_axis_off()
# fig.tight_layout()
# plt.show()


'''
create SVM classifier
'''

'''
ValueError: setting an array element with a sequence.

sift des returns a 1d array and i would like to change to 2d for it to work, but no effective method found yet
'''

# from train_SVM import train_linear_SVM
# Create a classifier: a support vector classifier
# classifier = svm.SVC(gamma=0.001)
classifier = svm.SVC(kernel='linear')

# Train the classifier
classifier.fit( X_train_SIFT_flattened ,y_train)


'''
Preict using SVM Classifier and test data
'''
# predict the classes on the images of the test set
y_pred = classifier.predict(X_test_SIFT_flattened)

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