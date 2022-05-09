
from base2 import LoadWildData, LoadTrainData
from joblib import dump, load
from sklearn import svm, metrics
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from keras.models import load_model
import cv2
import numpy as np



'''
Normal data
'''


X_test,y_test=LoadWildData(4)

# print("X_test ",X_test.shape)
# print("y_test ", y_test.shape)

# fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5), sharex=True, sharey=True)
# ax = axes.ravel()

# for i in range(10):
#     ax[i].imshow(X_test[i], cmap='gray')
#     ax[i].set_title(f'Label: {y_test[i]}')
#     ax[i].set_axis_off()
# fig.tight_layout()
# plt.show()




# # CNN
# file_name = 'CNN2.h5'
# checkpoint_path = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/Models/'+file_name
# print("loading ", file_name)
# cnn_classifier = load_model(checkpoint_path)





# '''
# Preict using CNN Classifier and wild data
# '''
# # predict the classes on the images of the test set
# y_pred = cnn_classifier.predict(X_test)
# # y_pred = int(np.argmax(y_pred[1]))
# # print(y_pred)

# X_test, y_test, y_pred = shuffle(X_test, y_test, y_pred)


# fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6), sharex=True, sharey=True)
# ax = axes.ravel()


# for i in range(10):
#     y_test_value = emotion_dict[y_test[i]]
#     y_pred_value = emotion_dict[int(np.argmax(y_pred[i]))]
#     ax[i].imshow(X_test[i], cmap='gray')
#     ax[i].set_title(f'Label: {y_test_value}\n Prediction: {y_pred_value}')
#     ax[i].set_axis_off()
# fig.tight_layout
# plt.show()



# # predict the classes on the images of the test set
# X_test, y_test = shuffle(X_test, y_test)

# score = cnn_classifier.evaluate(X_test, y_test, verbose=0)
# print('test loss: {:.2f}'.format(score[0]))
# print('test acc: {:.2f}'.format(score[1]))
# print('score: ',score)

