'''
Load train & test data 
'''

# import the necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_ubyte, img_as_float,data, exposure
from skimage.feature import hog
import tensorflow as tf
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.utils import shuffle
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from sklearn.cluster import MiniBatchKMeans

print("Running base.py")
print("Loading dataset")
DATASET_DIR = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/CW_Dataset/'



def LoadDataHOG(_testSize):
    # load the train & test data
    df_train = pd.read_csv(DATASET_DIR + '/labels/list_label_train.txt', sep=" ",  header=None, dtype=None)
    df_test = pd.read_csv(DATASET_DIR + '/labels/list_label_test.txt', sep=" ",  header=None, dtype=None)

    # name the colums
    df_train.columns = ["name", "label"]
    df_test.columns = ["name", "label"]


    # store the image array an label in a 1d array 
    # train arrays
    train_image = []
    train_image_HOG = []
    train_image_data = []
    # test arrays
    test_image = []
    test_image_HOG = []
    test_image_data = []
    
    # sample size
    # testSize = df_train.shape[0] #all the data
    testSize= _testSize #first 100 rows


    # shorten the train data for testing

    df_train = df_train[:testSize]
    df_test = df_test[:testSize]

    '''
    load train data images associating it with the dataframe above
    '''
    print(df_train)

    # loop through the csv files, using the file name to load the image and store in trian_image array
    for i in tqdm(range(df_train.shape[0])):
        # get the file name,remove the .jpg and store in val
        val = df_train['name'][i]
        val = val.split('.')[0] 
        # get the target value assiciated with the filename
        target = df_train['label'][i]

        # load image based on the filename colelcted from the csv
        img = image.load_img(DATASET_DIR +'/train/'+val+'_aligned.jpg', target_size=(100,100,1))

        # apply HOG feature descriptor
        HOG_des, HOG_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)

        # # Rescale histogram for better display
        HOG_image_rescaled = exposure.rescale_intensity(HOG_image, in_range=(0, 10))

        # convert the given image into  numpy array
        img = image.img_to_array(img)
        # Rescale the intensities
        img = img/255
        train_image.append(img)
        train_image_HOG.append(HOG_image_rescaled)
        train_image_data.append(target)

    X_train = np.array(train_image)
    X_train_HOG = np.array(train_image_HOG)
    X_train_HOG_flattened = X_train_HOG.reshape(len(X_train_HOG),-1)
    y_train = np.array(train_image_data)
    print("X_train ",X_train.shape)
    print("X_train_HOG ",X_train_HOG.shape)
    print("X_train_HOG_flattened ",X_train_HOG_flattened.shape)
    print("y_train ", y_train.shape)

    '''
    load test data images associating it with the dataframe above
    '''
    print(df_test)

    # loop through the csv files, using the file name to laod the image and store in test_image array
    for i in tqdm(range(df_test.shape[0])):

        # get the file name,remove the .jpg and store in val
        val_test = df_test['name'][i]
        val_test = val_test.split('.')[0] 

        # get the target value assiciated with the filename
        target_test = df_test['label'][i]
        # print(val, ":", target)

        # load image based on the filename colelcted from the csv
        img_test = image.load_img(DATASET_DIR +'/test/'+val_test+'_aligned.jpg', target_size=(100,100,1))

        # apply HOG feature descriptor
        HOG_des, HOG_image_test = hog(img_test, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)

        # # Rescale histogram for better display
        HOG_image_rescaled_test = exposure.rescale_intensity(HOG_image_test, in_range=(0, 10))

        # convert the given image into  numpy array
        img_test = image.img_to_array(img_test) #print(type(img_numpy_array))
        # Rescale the intensities
        img_test = img_test/255
        test_image.append(img_test)
        test_image_data.append(target_test)
        test_image_HOG.append(HOG_image_rescaled_test)

    X_test = np.array(test_image)
    X_test_HOG = np.array(test_image_HOG)
    X_test_HOG_flattened= X_test_HOG.reshape(len(X_test_HOG),-1)
    y_test = np.array(test_image_data)
    print("X_test ",X_test.shape)
    print("X_test_HOG ",X_test_HOG.shape)
    print("X_test_HOG_flattened ",X_test_HOG_flattened.shape)
    print("y_test ", y_test.shape)

    return X_train,X_train_HOG,X_train_HOG_flattened,y_train,X_test,X_test_HOG,X_test_HOG_flattened,y_test


'''
Load Data using SIFT
'''
def LoadDataSIFT(_testSize):
    # load the train & test data
    df_train = pd.read_csv(DATASET_DIR + '/labels/list_label_train.txt', sep=" ",  header=None, dtype=None)
    df_test = pd.read_csv(DATASET_DIR + '/labels/list_label_test.txt', sep=" ",  header=None, dtype=None)

    # name the colums
    df_train.columns = ["name", "label"]
    df_test.columns = ["name", "label"]


    # store the image array an label in a 1d array 
    # train arrays
    train_image = []
    train_image_SIFT = []
    train_image_data = []
    # test arrays
    test_image = []
    test_image_SIFT = []
    test_image_data = []

    # sample size
    # testSize = df_train.shape[0] #all the data
    testSize= _testSize #first 100 rows


    # shorten the train data for testing
    df_train = df_train[:testSize]
    df_test = df_test[:testSize]

    '''
    load train data images associating it with the dataframe above
    '''
    print(df_train)
    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # Create empty lists for feature descriptors and labels
    des_list = []
    y_train_list = []
        
    # loop through the csv files, using the file name to load the image and store in trian_image array
    for i in tqdm(range(df_train.shape[0])):
        # get the file name,remove the .jpg and store in val
        val = df_train['name'][i]
        val = val.split('.')[0] 
        # get the target value assiciated with the filename
        target = df_train['label'][i]
        # load image based on the filename colelcted from the csv
        img = image.load_img(DATASET_DIR +'/train/'+val+'_aligned.jpg', target_size=(100,100,1))

        # Identify keypoints and extract descriptors with SIFT
        img = img_as_ubyte(color.rgb2gray(X_train[i]))
        kp, des = sift.detectAndCompute(img, None)

        # Identify the keypoints and compute the descriptors with SIFT
        kp1, des = sift.detectAndCompute(img_as_ubyte(img), None)
        # kp1 list of coordiantes and des list of points (not image!!) thats why it doesnt work array of 1d arrays
        # Show results for first 4 images
        
        if i<4:
            img_with_SIFT = cv2.drawKeypoints(img_as_ubyte(img), kp1, img)
            ax[i].imshow(img_with_SIFT)
            ax[i].set_axis_off()
        
            # Append list of descriptors and label to respective lists
        if des is not None:
            train_image_SIFT.append(des)
        # convert the given image into  numpy array
        img = image.img_to_array(img)
        # Rescale the intensities
        img = img/255
        train_image.append(img)
        train_image_data.append(target)

    des_array = np.vstack(train_image_SIFT)
    # Number of centroids/codewords: good rule of thumb is 10*num_classes
    k = len(np.unique(y_train)) * 10

    # Use MiniBatchKMeans for faster computation and lower memory usage
    batch_size = des_array.shape[0] // 4
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size).fit(des_array)

    # Convert descriptors into histograms of codewords for each image
    hist_list = []
    idx_list = []

    for des in train_image_SIFT:
        hist = np.zeros(k)

        idx = kmeans.predict(des)
        idx_list.append(idx)
        for j in idx:
            hist[j] = hist[j] + (1 / len(des))
        hist_list.append(hist)

    train_hist_array = np.vstack(hist_list)

    X_train = np.array(train_image)
    X_train_SIFT = np.array(train_image_SIFT)
    X_train_SIFT_flattened = X_train_SIFT.reshape(len(X_train_SIFT),-1)
    y_train = np.array(train_image_data)
    print("X_train ",X_train.shape)
    print("X_train_SIFT ",X_train_SIFT.shape)
    print("X_train_hist_array  ",train_hist_array .shape)
    print("X_train_SIFT_flattened ",X_train_SIFT_flattened.shape)
    print("y_train ", y_train.shape)

    '''
    load test data images associating it with the dataframe above
    '''
    print(df_test)

    # loop through the csv files, using the file name to laod the image and store in test_image array
    for i in tqdm(range(df_test.shape[0])):

        # get the file name,remove the .jpg and store in val
        val_test = df_test['name'][i]
        val_test = val_test.split('.')[0] 

        # get the target value assiciated with the filename
        target_test = df_test['label'][i]
        # print(val, ":", target)

        # load image based on the filename colelcted from the csv
        img_test = image.load_img(DATASET_DIR +'/test/'+val_test+'_aligned.jpg', target_size=(100,100,1))


        # Identify the keypoints and compute the descriptors with SIFT
        kp1, des_test = sift.detectAndCompute(img_as_ubyte(img_test), None)

        # convert the given image into  numpy array
        img_test = image.img_to_array(img_test) #print(type(img_numpy_array))
        # Rescale the intensities
        img_test = img_test/255
        test_image.append(img_test)
        test_image_data.append(target_test)
        test_image_SIFT.append(des_test)

    X_test = np.array(test_image)
    X_test_SIFT = np.array(test_image_SIFT)
    X_test_SIFT_flattened= X_test_SIFT.reshape(len(X_test_SIFT),-1)
    y_test = np.array(test_image_data)
    print("X_test ",X_test.shape)
    print("X_test_SIFT ",X_test_SIFT.shape)
    print("X_test_SIFT_flattened ",X_test_SIFT_flattened.shape)
    print("y_test ", y_test.shape)

    return X_train,X_train_SIFT,X_train_SIFT_flattened,y_train,X_test,X_test_SIFT,X_test_SIFT_flattened,y_test


'''
Load data normally
'''
def LoadData(_testSize):
    # load the train & test data
    df_train = pd.read_csv(DATASET_DIR + '/labels/list_label_train.txt', sep=" ",  header=None, dtype=None)
    df_test = pd.read_csv(DATASET_DIR + '/labels/list_label_test.txt', sep=" ",  header=None, dtype=None)

    # name the colums
    df_train.columns = ["name", "label"]
    df_test.columns = ["name", "label"]


    # store the image array an label in a 1d array 
    # train arrays
    train_image = []
    train_image_data = []
    # test arrays
    test_image = []
    test_image_data = []
    
    # sample size
    # testSize = df_train.shape[0] #all the data
    testSize= _testSize #first 100 rows


    # shorten the train data for testing
    df_train = df_train[:testSize]
    df_test = df_test[:testSize]

    # if _testSize > df_test.length():

    '''
    load train data images associating it with the dataframe above
    '''
    print(df_train)

    # loop through the csv files, using the file name to load the image and store in trian_image array
    for i in tqdm(range(df_train.shape[0])):
        # get the file name,remove the .jpg and store in val
        val = df_train['name'][i]
        val = val.split('.')[0] 
        # get the target value assiciated with the filename
        target = df_train['label'][i]

        # load image based on the filename colelcted from the csv
        img = image.load_img(DATASET_DIR +'/train/'+val+'_aligned.jpg', target_size=(100,100,1))

        # convert the given image into  numpy array
        img = image.img_to_array(img)
        # Rescale the intensities
        img = img/255
        

        train_image.append(img)
        train_image_data.append(target)

    X_train = np.array(train_image)
    X_train_flattened = X_train.reshape(len(X_train),-1)
    y_train = np.array(train_image_data)
    print("X_train ",X_train.shape)
    print("y_train ", y_train.shape)

    '''
    load test data images associating it with the dataframe above
    '''
    print(df_test)

    # loop through the csv files, using the file name to laod the image and store in test_image array
    for i in tqdm(range(df_test.shape[0])):

        # get the file name,remove the .jpg and store in val
        val_test = df_test['name'][i]
        val_test = val_test.split('.')[0] 

        # get the target value assiciated with the filename
        target_test = df_test['label'][i]
        # print(val, ":", target)

        # load image based on the filename colelcted from the csv
        img_test = image.load_img(DATASET_DIR +'/test/'+val_test+'_aligned.jpg', target_size=(100,100,1))

        # convert the given image into  numpy array
        img_test = image.img_to_array(img_test) #print(type(img_numpy_array))
        # Rescale the intensities
        img_test = img_test/255
        test_image.append(img_test)
        test_image_data.append(target_test)

    X_test = np.array(test_image)
    X_test_flattened = X_test.reshape(len(X_test),-1)
    y_test = np.array(test_image_data)
    print("X_test ",X_test.shape)
    print("y_test ", y_test.shape)


    return X_train,X_train_flattened,y_train,X_test,X_test_flattened,y_test


