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
Normal data
'''
X_NORM,X_NORM_flattened,y_NORM = LoadTrainData('all')
# X_NORM,X_NORM_flattened,y_NORM = LoadTrainData(100)

print("X_train ",X_NORM.shape)
print("X_train_flattened ",X_NORM_flattened.shape)
print("y_train ", y_NORM.shape)

# # SVM
# # create SVM classifier
# # Create a classifier: a support vector classifier
# classifier = svm.SVC(kernel='linear')

# # Train the classifier
# classifier.fit( X_NORM_flattened,y_NORM)

# print("Dumping file")
# file_name = 'SVM.joblib'
# checkpoint_path = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/Models/'+file_name
# dump(classifier, checkpoint_path) 

# # CNN
# function to the train the datasets, sample(used for testing purposes)  and actual
X_train, X_validate, y_train, y_validate = train_test_split(X_NORM, y_NORM, test_size = 0.2, random_state = 1)


classifier = Sequential()
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

# Compile the model
classifier.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])# metrics = What you want to maximise

history= classifier.fit(X_train, y_train,epochs=40,validation_data=(X_validate, y_validate))

print("\n-----------------------------CNN model accuracy (Validation test) ----------------------------------")
score = classifier.evaluate(X_validate, y_validate, verbose=0)

print('validation loss: {:.2f}'.format(score[0]))
print('validation acc: {:.2f}'.format(score[1]))

print("Dumping file...")
file_name = 'CNN2.h5'
checkpoint_path = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/Models/'+file_name
classifier.save(checkpoint_path)
print("model saved!")



# MLP
# X_train, X_validate, y_train, y_validate = train_test_split(X_NORM, y_NORM, test_size = 0.2, random_state = 1)

# classifier = MLPClassifier(
#                     hidden_layer_sizes=(50),
#                     alpha=1e-4,
#                     solver='sgd',
#                     # max_iter=100,
#                     verbose=True,
#                     learning_rate_init=0.001,
#                     random_state=1
#                     )
# classifier.fit(X_NORM_flattened,y_NORM)

# print("Dumping file...")          
# file_name = 'MLP.joblib'
# checkpoint_path = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/Models/'+file_name

# dump(classifier, checkpoint_path) 
# print("model saved!")



'''
HOG
'''
# X_HOG_img,X_HOG,X_HOG_flattened,y_HOG =LoadTrainDataHOG('all')
# # X_HOG_img,X_HOG,X_HOG_flattened,y_HOG =LoadTrainDataHOG(100)
# print("X_train_img ",X_HOG_img.shape)
# print("X_train ",X_HOG.shape)
# print("X_train_flattened ",X_HOG_flattened.shape)
# print("y_train ", y_HOG.shape)

# # SVM
# # create SVM classifier
# # Create a classifier: a support vector classifier
# classifier = svm.SVC(kernel='linear')

# # Train the classifier
# classifier.fit(X_HOG_flattened, y_HOG)

# print("Dumping file")
# file_name = 'HOG-SVM.joblib'
# checkpoint_path = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/Models/'+file_name
# dump(classifier, checkpoint_path) 


# # MLP
# # X_train, X_validate, y_train, y_validate = train_test_split(X_HOG, y_HOG, test_size = 0.2, random_state = 1)

# classifier = MLPClassifier(
#                     hidden_layer_sizes=(50),
#                     alpha=1e-4,
#                     solver='sgd',
#                     # max_iter=100,
#                     verbose=True,
#                     learning_rate_init=0.001,
#                     random_state=1
#                     )
# classifier.fit(X_HOG_flattened,y_HOG)

# print("Dumping file...")          
# file_name = 'HOG-MLP.joblib'
# checkpoint_path = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/Models/'+file_name

# dump(classifier, checkpoint_path) 
# print("model saved!")



'''
SIFT
'''
# X_SIFT,X_SIFT_flattened,y_SIFT =LoadTrainDataSIFT(100)
# print("X_train ",X_SIFT.shape)
# print("X_train_flattened ",X_SIFT_flattened.shape)
# print("y_train ", y_SIFT.shape)

# # MLP
# # X_train, X_validate, y_train, y_validate = train_test_split(X_NORM, y_NORM, test_size = 0.2, random_state = 1)

# classifier = MLPClassifier(
#                     hidden_layer_sizes=(50),
#                     alpha=1e-4,
#                     solver='sgd',
#                     # max_iter=100,
#                     verbose=True,
#                     learning_rate_init=0.001,
#                     random_state=1
#                     )
# classifier.fit(X_SIFT_flattened,y_SIFT)

# print("Dumping file...")          
# file_name = 'SIFT-MLP.joblib'
# checkpoint_path = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/Models/'+file_name

# dump(classifier, checkpoint_path) 
# print("model saved!")

