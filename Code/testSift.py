# import the necessary libraries
import os
from coursework.CW_Folder_UG.Code.base2 import LoadTestData, LoadTrainData
import cv2
from sklearn.model_selection import train_test_split
from skimage import img_as_ubyte, io, color
from sklearn.cluster import MiniBatchKMeans
from sklearn import svm, metrics
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import tensorflow as tf
from skimage import color
from tqdm import tqdm
from base2 import LoadData

DATASET_DIR = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/CW_Dataset/'


def LoadDataSIFT(_testSize):
    X_train,X_train_flattened,y_train = LoadTrainData(_testSize)
    X_test, X_test_flattened,y_test = LoadTestData(_testSize)

    #function to the train the datasets, sample(used for testing purposes)  and actual
    # X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = 0.2, shuffle=True,stratify=y_train)
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, stratify=y_train)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # Create empty lists for feature descriptors and labels
    des_list = []
    y_train_list = []

    fig, ax = plt.subplots(1, 4, figsize=(10, 8), sharey=True)

    for i in tqdm(range(len(X_train))):
        # Identify keypoints and extract descriptors with SIFT
        img = img_as_ubyte(color.rgb2gray(X_train[i]))
        kp, des = sift.detectAndCompute(img, None)

            
        # Append list of descriptors and label to respective lists
        if des is not None:
            des_list.append(des)
            y_train_list.append(y_train[i])


    # Convert to array for easier handling
    des_array = np.vstack(des_list)
    # des_array:  (107645, 128)

    # Number of centroids/codewords: good rule of thumb is 10*num_classes
    k = len(np.unique(y_train)) * 10
    # Use MiniBatchKMeans for faster computation and lower memory usage
    batch_size = des_array.shape[0] // 4
    kmeans = MiniBatchKMeans(n_clusters=k,batch_size=batch_size).fit(des_array)

    print("des_array: ", des_array.shape)
    print ("k: ", k)
    print ("batch_size: ", batch_size)


    # Convert descriptors into histograms of codewords for each image
    hist_list = []
    idx_list = []

    for des in des_list:
        hist = np.zeros(k)

        idx = kmeans.predict(des)
        idx_list.append(idx)
        for j in idx:
            hist[j] = hist[j] + (1 / len(des))
        hist_list.append(hist)

    hist_array_train = np.vstack(hist_list)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(np.array(idx_list, dtype=object), bins=k)
    ax.set_title('Codewords occurrence in training set')
    plt.show()

    hist_list_test = []

    print("X_train ",hist_array_train.shape)
    print("y_train ", y_train.shape)

    # This concludes the training of the BoVW model.
    # We can now move to the test set and estimate the accuracy of the model. 
    # we need to detect interest points, extract feature descriptors and associate histograms of codewords from each image of the test set
    for i in range(len(X_test)):
        img = img_as_ubyte(color.rgb2gray(X_test[i]))
        kp, des = sift.detectAndCompute(img, None)

        if des is not None:
            hist = np.zeros(k)

            idx = kmeans.predict(des)

            for j in idx:
                hist[j] = hist[j] + (1 / len(des))

            # hist = scale.transform(hist.reshape(1, -1))
            hist_list_test.append(hist)

        else:
            hist_list_test.append(None)

    # Remove potential cases of images with no descriptors
    # idx_not_empty = [i for i, x in enumerate(hist_list) if x is not None]
    # hist_list = [hist_list[i] for i in idx_not_empty]
    # y_test = [y_test[i] for i in idx_not_empty]

    hist_array_test = np.vstack(hist_list_test)

    print("X_train ",hist_array_test.shape)
    print("y_train ", y_test.shape)

    return hist_array_train,y_train, hist_array_test,y_test