'''
Load train & test data 
'''

# import the necessary libraries
from ast import If
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
from keras.models import load_model


print("Running base.py")
print("Loading dataset")
DATASET_DIR = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/CW_Dataset/'

'''
HOG
'''
def LoadTrainDataHOG(_testSize):
    # load the train & test data
    df_train = pd.read_csv(DATASET_DIR + '/labels/list_label_train.txt', sep=" ",  header=None, dtype=None)

    # name the colums
    df_train.columns = ["name", "label"]


    # store the image array an label in a 1d array 
    # train arrays
    train_image = []
    train_image_HOG = []
    train_image_data = []
    
    if (_testSize != 'all'):
        df_train = df_train[:_testSize] #first x rows 
    
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

    return X_train,X_train_HOG,X_train_HOG_flattened,y_train

def LoadTestDataHOG(_testSize):
    # load the train & test data
    df_test = pd.read_csv(DATASET_DIR + '/labels/list_label_test.txt', sep=" ",  header=None, dtype=None)

    # name the colums
    df_test.columns = ["name", "label"]


    # store the image array an label in a 1d array 
    # test arrays
    test_image = []
    test_image_HOG = []
    test_image_data = []
    
    if (_testSize != 'all'):
        df_test = df_test[:_testSize] #first x rows 


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

    return X_test,X_test_HOG,X_test_HOG_flattened,y_test


'''
Load data normally
'''
def LoadTrainData(_testSize):
    # load the train & test data
    df_train = pd.read_csv(DATASET_DIR + '/labels/list_label_train.txt', sep=" ",  header=None, dtype=None)

    # name the colums
    df_train.columns = ["name", "label"]

    # store the image array an label in a 1d array 
    # train arrays
    train_image = []
    train_image_data = []
 
    
    if (_testSize != 'all'):
        df_train = df_train[:_testSize] #first x rows 



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

    return X_train,X_train_flattened,y_train


'''
Load data normally
'''
def LoadTestData(_testSize):
    # load the train & test data
    df_test = pd.read_csv(DATASET_DIR + '/labels/list_label_test.txt', sep=" ",  header=None, dtype=None)

    # name the colums
    df_test.columns = ["name", "label"]


    # store the image array an label in a 1d array 
    # test arrays
    test_image = []
    test_image_data = []
    
      
    if (_testSize != 'all'):
        df_test = df_test[:_testSize] #first x rows 


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


    return X_test,X_test_flattened,y_test



'''
SIFT
'''

def LoadTrainDataSIFT(_testSize):
    X_train,X_train_flattened,y_train = LoadTrainData(_testSize)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # Create empty lists for feature descriptors and labels
    des_list = []
    y_train_list = []
    
    fig, ax = plt.subplots(1, 4, figsize=(10, 8), sharey=True)

    # loop through the csv files, using the file name to load the image and store in trian_image array
    for i in tqdm(range(len(X_train))):
        # Identify keypoints and extract descriptors with SIFT
        img = img_as_ubyte(color.rgb2gray(X_train[i]))
        kp, des = sift.detectAndCompute(img, None)

        # Show results for first 4 images
        if i<4:
            img_with_SIFT = cv2.drawKeypoints(img, kp, img)
            ax[i].imshow(img_with_SIFT)
            ax[i].set_axis_off()
            
        # Append list of descriptors and label to respective lists
        if des is not None:
            des_list.append(des)
            y_train_list.append(y_train[i])

    fig.tight_layout()
    plt.show()

    # Convert to array for easier handling
    des_array = np.vstack(des_list)


    print("des_array: ", des_array.shape)

    des_array_flattened = des_array.reshape(len(des_array),-1)

    print("X_train ",des_array.shape)
    print("X_train_SIFT_flattened ",des_array_flattened.shape)
    # print("y_train ", y_train_list.shape)

    return des_array,des_array_flattened,y_train


def LoadTestDataSIFT(_testSize):
    X_test,X_test_flattened,y_test = LoadTestData(_testSize)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # Create empty lists for feature descriptors and labels
    des_list = []
    y_test_list = []
    
    fig, ax = plt.subplots(1, 4, figsize=(10, 8), sharey=True)

    # loop through the csv files, using the file name to load the image and store in trian_image array
    for i in tqdm(range(len(X_test))):
        # Identify keypoints and extract descriptors with SIFT
        img = img_as_ubyte(color.rgb2gray(X_test[i]))
        kp, des = sift.detectAndCompute(img, None)

        # Show results for first 4 images
        if i<4:
            img_with_SIFT = cv2.drawKeypoints(img, kp, img)
            ax[i].imshow(img_with_SIFT)
            ax[i].set_axis_off()
            
        # Append list of descriptors and label to respective lists
        if des is not None:
            des_list.append(des)
            y_test_list.append(y_test[i])

    fig.tight_layout()
    plt.show()

    # Convert to array for easier handling
    des_array = np.vstack(des_list)

    print("des_array: ", des_array.shape)
  

    des_array_flattened = des_array.reshape(len(des_array),-1)

    print("X_train ",des_array.shape)
    print("X_train_SIFT_flattened ",des_array_flattened.shape)
    # print("y_train ", y_test_list.shape)

    return des_array,des_array_flattened,y_test




##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################


# def LoadDataSIFT(_testSize):


#     X_train,X_train_flattened,y_train = LoadTrainData(_testSize)

#     # Initiate SIFT detector
#     sift = cv2.SIFT_create()

#     # Create empty lists for feature descriptors and labels
#     des_list = []
#     y_train_list = []
    
#     fig, ax = plt.subplots(1, 4, figsize=(10, 8), sharey=True)

#     # loop through the csv files, using the file name to load the image and store in trian_image array
#     for i in tqdm(range(len(X_train))):
#         # Identify keypoints and extract descriptors with SIFT
#         img = img_as_ubyte(color.rgb2gray(X_train[i]))
#         kp, des = sift.detectAndCompute(img, None)
    
#         # extract the interest points using a fixed grid, for better accuracy
#         sift.compute(img, kp)

#         # Show results for first 4 images
#         if i<4:
#             img_with_SIFT = cv2.drawKeypoints(img, kp, img)
#             ax[i].imshow(img_with_SIFT)
#             ax[i].set_axis_off()
            
#         # Append list of descriptors and label to respective lists
#         if des is not None:
#             des_list.append(des)
#             y_train_list.append(y_train[i])

#     fig.tight_layout()
#     plt.show()

#     # Convert to array for easier handling
#     des_array = np.vstack(des_list)

#     # extract the interest points using a fixed grid, for better accuracy
#     # kp = cv2.KeyPoint

#     # Number of centroids/codewords: good rule of thumb is 10*num_classes
#     k = len(np.unique(y_train)) * 10

#     # Use MiniBatchKMeans for faster computation and lower memory usage
#     batch_size = des_array.shape[0] // 4

#     kmeans = MiniBatchKMeans(n_clusters=k,batch_size=batch_size)
#     kmeans.fit(des_array)

#     print("des_array: ", des_array.shape)
#     print ("k: ", k)
#     print ("batch_size: ", batch_size)
   
#    # Convert descriptors into histograms of codewords for each image
#     hist_list = []
#     idx_list = []

#     for des in des_list:
#         hist = np.zeros(k)

#         idx = kmeans.predict(des)
#         idx_list.append(idx)
#         for j in idx:
#             hist[j] = hist[j] + (1 / len(des))
#         hist_list.append(hist)

#     hist_array_train = np.vstack(hist_list)

#     fig, ax = plt.subplots(figsize=(8, 3))
#     ax.hist(np.array(idx_list, dtype=object), bins=k)
#     ax.set_title('Codewords occurrence in training set')
#     plt.show()


#     hist_array_flattened_train = hist_array_train.reshape(len(hist_array_train),-1)

#     print("X_train ",hist_array_train.shape)
#     print("X_train_SIFT_flattened ",hist_array_flattened_train.shape)
#     print("y_train ", y_train.shape)

#     X_test,X_test_flattened,y_test = LoadTestData(_testSize)


#    # Convert descriptors into histograms of codewords for each image
#     hist_list = []


#     for i in range(len(X_test)):
#         img = img_as_ubyte(color.rgb2gray(X_test[i]))
#         kp, des = sift.detectAndCompute(img, None)

#         if des is not None:
#             hist = np.zeros(k)

#             idx = kmeans.predict(des)

#             for j in idx:
#                 hist[j] = hist[j] + (1 / len(des))

#             # hist = scale.transform(hist.reshape(1, -1))
#             hist_list.append(hist)

#         else:
#             hist_list.append(None)

#     # Remove potential cases of images with no descriptors
#     idx_not_empty = [i for i, x in enumerate(hist_list) if x is not None]
#     hist_list = [hist_list[i] for i in idx_not_empty]
#     y_test = [y_test[i] for i in idx_not_empty]
#     hist_array_test = np.vstack(hist_list)

#     hist_array_flattened_test = hist_array_test.reshape(len(hist_array_test),-1)

#     print("X_train ",hist_array_test.shape)
#     print("X_train_SIFT_flattened ",hist_array_flattened_test.shape)
#     # print("y_train ", y_test_list.shape)

#     return hist_array_train,hist_array_flattened_train,y_train,hist_array_test,hist_array_flattened_test,y_test


# def LoadWildData():

#     emotion_dict ={1: "Surprise",2: "Fear",3: "Disgust",4: "Happiness",5: "Sadness",6: "Anger",7: "Neutral"}
#     # Load the cascade 
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     file_name = 'CNN2.h5'
#     checkpoint_path = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/Models/'+file_name
#     print("loading ", file_name)
#     cnn= load_model(checkpoint_path)
 
#     # crate window for better vieing experice
#     # cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#     # cv2.resizeWindow('image', 600,600) 


#     # WILD_DATA_DIR  = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/Personal_Dataset/'
#     WILD_DATA_DIR  = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/Personal_Dataset'

#     # store all files  train directory in a variable called files
#     files = os.listdir(WILD_DATA_DIR)

#     images = []
#     labels = []

#     # loop through all directory storing each dir name as label
#     for filename in files:
#         if (filename ==".DS_Store"):
#             # print(filename, "ignored")
#             continue
#         else:
#             # print(filename)
#             label = filename
#             EMOTION_DIR = WILD_DATA_DIR+"/"+filename
#             images = os.listdir(EMOTION_DIR)
#             # loop through all images and manipulate them and store in array
#             for imageFile in images:
#                 if (imageFile ==".DS_Store"):
#                     continue
#                 else:
#                     # print(imageFile)
#                     INDV_DIR= EMOTION_DIR +'/'+imageFile
#                     print(INDV_DIR)
#                     # load image 
#                     # img_test = image.load_img(INDV_DIR, target_size=(100,100,1))
#                     img_test = cv2.imread(INDV_DIR)

#                     width = 100
#                     height = 100
#                     dim = (width, height)
#                     img_resized = cv2.resize(img_test,dim, interpolation = cv2.INTER_AREA)




#                     # print(img_resized.shape)
#                     # Convert to grayscale 
#                     # img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

#                     # Detect the faces 
#                     faces = face_cascade.detectMultiScale(img_test, 1.1, 4)

#                     # crop the fram to show the face only
#                     # faces = faces[:10]
#                     for (x,y,w,h) in faces:
#                         cv2.rectangle(img_test,(x,y), (x+w,y+h), (255,0,0),2)
#                         roi_grey_img = img_test[y:y+h, x:x+w]
#                         cropped_img = np.expand_dims(cv2.resize(roi_grey_img,(100,100)),0)


#                         # preditct emotions
#                         y_pred = cnn.predict(cropped_img)
#                         maxindex = int(np.argmax(y_pred))
#                         cv2.putText(roi_grey_img,emotion_dict[maxindex],(x+5,y-20), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 2, cv2.LINE_AA)

#                         cv2.imshow("IMG",roi_grey_img )
#                         cv2.waitKey(0)

#                         # closing all open windows
#                         cv2.destroyAllWindows()


#                     # convert the given image into  numpy array
#                     # img_test = image.img_to_array(img_test) #print(type(img_numpy_array))
#                     # # Rescale the intensities
#                     # img_test = img_test/255
#                     # images.append(img_test)
#                     # labels.append(label)

#                     # X_test = np.array(images)
#                     # y_test = np.array(labels)
#                     # print("X_test ",X_test.shape)
#                     # print("y_test ", y_test.shape)


#                     # return X_test,y_test

  


# def LoadWildData():

#     emotion_dict ={1: "Surprise",2: "Fear",3: "Disgust",4: "Happiness",5: "Sadness",6: "Anger",7: "Neutral"}
#     # Load the cascade 
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     file_name = 'CNN2.h5'
#     checkpoint_path = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/Models/'+file_name
#     print("loading ", file_name)
#     cnn= load_model(checkpoint_path)
 
#     # crate window for better vieing experice
#     # cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#     # cv2.resizeWindow('image', 600,600) 


#     # WILD_DATA_DIR  = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/Personal_Dataset/'
#     WILD_DATA_DIR  = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/Personal_Dataset'

#     # store all files  train directory in a variable called files
#     files = os.listdir(WILD_DATA_DIR)

#     images = []
#     labels = []

#     # loop through all directory storing each dir name as label
#     for filename in files:
#         if (filename ==".DS_Store"):
#             # print(filename, "ignored")
#             continue
#         else:
#             # print(filename)
#             label = filename
#             EMOTION_DIR = WILD_DATA_DIR+"/"+filename
#             images = os.listdir(EMOTION_DIR)
#             # loop through all images and manipulate them and store in array
#             for imageFile in images:
#                 if (imageFile ==".DS_Store"):
#                     continue
#                 else:
#                     # print(imageFile)
#                     INDV_DIR= EMOTION_DIR +'/'+imageFile
#                     print(INDV_DIR)
#                     # load image 
#                     # img_test = image.load_img(INDV_DIR, target_size=(100,100,1))
#                     img_test = cv2.imread(INDV_DIR)

#                     width = 100
#                     height = 100
#                     dim = (width, height)
#                     img_resized = cv2.resize(img_test,dim, interpolation = cv2.INTER_AREA)
#                     cropped_img = np.expand_dims(cv2.resize(img_resized,(100,100)),0)


#                     labels.append(labels)

#     # convert the given image into  numpy array
#     img_test = image.img_to_array(img_test) #print(type(img_numpy_array))
                
#     # Rescale the intensities
#     img_test = img_test/255
#     images.append(img_test)
#     labels.append(label)

#     X_test = np.array(images)
#     y_test = np.array(labels)
#     print("X_test ",X_test.shape)
#     print("y_test ", y_test.shape)


#     # return X_test,y_test








# from skimage.transform import rescale, resize, downscale_local_mean

# def LoadWildData():
#     # WILD_DATA_DIR  = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/Personal_Dataset/'
#     WILD_DATA_DIR  = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/Personal_Dataset'

#     # store all files  train directory in a variable called files
#     files = os.listdir(WILD_DATA_DIR)

#     images = []
#     labels = []

#     # loop through all directory storing each dir name as label
#     for filename in files:
#         images_2 = []
#         labels_2 = []
#         if (filename ==".DS_Store"):
#             continue
#         else:
#             label = filename
#             EMOTION_DIR = WILD_DATA_DIR+"/"+filename
#             images = os.listdir(EMOTION_DIR)
#             # loop through all images and manipulate them and store in array
#             for imageFile in images:
#                 img = loadImage(filename, imageFile)
#                 images_2.append(img)
#                 labels_2.append(label)
#         images = np.array(images_2)
#         labels = np.array(labels_2)

#     X_train = np.array(images)
#     y_train = np.array(labels)
#     print("X_train ",X_train.shape)
#     print("y_train ", y_train.shape)

#     return X_train,y_train


# def loadImage(_filename,_imageFile):
#     EMOTION_DIR = WILD_DATA_DIR+"/"+_filename
#     if (_imageFile == ".DS_Store"):
#         return []
#     else:
#         INDV_DIR= EMOTION_DIR +'/'+_imageFile
#         print(INDV_DIR)
#         img = io.imread(INDV_DIR)

#         img = resize(img, (100, 100),
#             anti_aliasing=True)

#         # plt.imshow(img )
#         # plt.show()

#         # convert the given image into  numpy array
#         img = image.img_to_array(img)
#         # Rescale the intensities
#         img = img/255

#         return img 



'''
Load data normally
'''
from sklearn.utils import shuffle

def LoadWildData(_testSize):
    # load the train & test data
    WILD_DATA_DIR  = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/Personal_Dataset'
    df = pd.read_csv(WILD_DATA_DIR + '/labels/list_label_test.txt', sep=" ",  header=None, dtype=None)

    # df_test = shuffle(df_test, random_state=1)
    # shuffle the DataFrame rows
    # df = df_test.sample(frac = 1)

    # name the colums
    df.columns = ["name", "label"]

    df = df.sample(frac=1).reset_index()


    # store the image array an label in a 1d array 
    # test arrays
    test_image = []
    test_image_data = []
    
      
    if (_testSize != 'all'):
        df = df[:_testSize] #first x rows 


    '''
    load test data images associating it with the dataframe above
    '''
    print(df)

    # loop through the csv files, using the file name to laod the image and store in test_image array
    for i in tqdm(range(df.shape[0])):

        print(df['name'][i])
        # get the file name
        val_test = df['name'][i]

        # get the label value assiciated with the filename
        label_test = df['label'][i]
        print(val_test, ":", label_test)

        # load image based on the filename colelcted from the csv
        img_test = image.load_img(WILD_DATA_DIR +'/test/'+val_test, target_size=(100,100,1))

        # convert the given image into  numpy array
        img_test = image.img_to_array(img_test) #print(type(img_numpy_array))
        # Rescale the intensities
        img_test = img_test/255
        test_image.append(img_test)
        test_image_data.append(label_test)

    X_test = np.array(test_image)
    y_test = np.array(test_image_data)
    print("X_test ",X_test.shape)
    print("y_test ", y_test.shape)


    return X_test,y_test


# def LoadWildDataOLD1():

#     emotion_dict ={1: "Surprise",2: "Fear",3: "Disgust",4: "Happiness",5: "Sadness",6: "Anger",7: "Neutral"}
#     # Load the cascade 
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     file_name = 'CNN2.h5'
#     checkpoint_path = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/Models/'+file_name
#     print("loading ", file_name)
#     cnn= load_model(checkpoint_path)

#     # crate window for better vieing experice
#     cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('image', 600,600) 


#     # WILD_DATA_DIR  = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/Personal_Dataset/'
#     WILD_DATA_DIR  = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/Personal_Dataset'

#     # store all files  train directory in a variable called files
#     files = os.listdir(WILD_DATA_DIR)

#     images = []
#     labels = []

#     # loop through all directory storing each dir name as label
#     for filename in files:
#         if (filename ==".DS_Store"):
#             # print(filename, "ignored")
#             continue
#         else:
#             # print(filename)
#             label = filename
#             print(label)
#             EMOTION_DIR = WILD_DATA_DIR+"/"+filename
#             images = os.listdir(EMOTION_DIR)
#             # loop through all images and manipulate them and store in array
#             for imageFile in images:
#                 if (imageFile == ".DS_Store"):
#                     continue
#                 else:
#                     # print(imageFile)
#                     INDV_DIR= EMOTION_DIR +'/'+imageFile
#                     print(INDV_DIR)
#                     # load image 
#                     # img_test = image.load_img(INDV_DIR, target_size=(100,100,1))
#                     img_test = io.imread(INDV_DIR)
#                     img_gray = color.rgb2gray(img_test)
#                     img_gray = img_as_ubyte(img_gray)         # Conversion required by OpenCV

#                     # Detect the faces 
#                     faces = face_cascade.detectMultiScale(img_gray, 1.1, 4)

#                     # crop the fram to show the face only
#                     # faces = faces[:10]
#                     for (x,y,w,h) in faces:
#                         cv2.rectangle(img_gray,(x,y), (x+w,y+h), (255,0,0),2)
#                         roi_grey_img = img_gray[y:y+h, x:x+w]
#                         cropped_img = np.expand_dims(cv2.resize(roi_grey_img,(100,100)),0)
#                         # plt.imshow(roi_grey_img )
#                         # plt.show()
#                         # images.append(roi_grey_img)

#                                 # preditct emotions
#                         y_pred = cnn.predict(cropped_img)
#                         maxindex = int(np.argmax(y_pred))
#                         cv2.putText(roi_grey_img,emotion_dict[maxindex],(x+5,y-20), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 2, cv2.LINE_AA)
#                         # labels.append(label)
#                         plt.imshow(roi_grey_img )
#                         plt.show()

                       


#     X_test = np.array(cropped_img)
#     # X_images = np.array(images)
#     y_test = np.array(labels)
#     print("X_test ",X_test.shape)
#     print("y_test ", y_test.shape)

#     fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5), sharex=True, sharey=True)
#     ax = axes.ravel()

#     for i in range(10):
#         ax[i].imshow(X_test[i], cmap='gray')
#         ax[i].set_title(f'Label: {y_test[i]}')
#         ax[i].set_axis_off()
#     fig.tight_layout()
#     plt.show()



#     return X_test,y_test

  
