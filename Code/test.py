from base2 import LoadTrainData, LoadTrainDataHOG ,LoadTrainDataSIFT
from joblib import dump, load
from sklearn import svm, metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout,BatchNormalization,Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.neural_network import MLPClassifier



'''
SIFT
'''
X_SIFT,X_SIFT_flattened,y_SIFT =LoadTrainDataSIFT(100)
print("X_train ",X_SIFT.shape)
print("X_train_flattened ",X_SIFT_flattened.shape)
print("y_train ", y_SIFT.shape)
