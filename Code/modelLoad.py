from base2 import LoadTestData, LoadTestDataHOG ,LoadTestDataSIFT
from joblib import dump, load
from sklearn import svm, metrics
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from keras.models import load_model

'''
Normal data
'''
# X_test, X_test_flattened,y_test = LoadTestData('all')
X_test, X_test_flattened,y_test = LoadTestData(4)

# # SVM
# file_name = 'SVM.joblib'
# checkpoint_path = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/Models/'+file_name
# print("loading ", file_name)
# svm_classifier = load(checkpoint_path)

# '''
# Preict using SVM Classifier and test data
# '''
# # predict the classes on the images of the test set
# y_pred = svm_classifier.predict(X_test_flattened)

# X_test, y_test, y_pred = shuffle(X_test, y_test, y_pred)
# X_test_img = X_test

# fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6), sharex=True, sharey=True)
# ax = axes.ravel()

# for i in range(4):
#     ax[i].imshow(X_test_img[i], cmap='gray')
#     ax[i].set_title(f'Label: {y_test[i]}\n Prediction: {y_pred[i]}')
#     ax[i].set_axis_off()
# fig.tight_layout
# plt.show()

# print(f"""Classification report for classifier {svm_classifier}:\n
#       {metrics.classification_report(y_test, y_pred)}""")



# CNN
file_name = 'CNN2.h5'
checkpoint_path = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/Models/'+file_name
print("loading ", file_name)
cnn_classifier = load_model(checkpoint_path)

# predict the classes on the images of the test set
X_test, y_test = shuffle(X_test, y_test)

score = cnn_classifier.evaluate(X_test, y_test, verbose=0)
print('test loss: {:.2f}'.format(score[0]))
print('test acc: {:.2f}'.format(score[1]))
print('score: ',score)


# # MLP
# file_name = 'MLP.joblib'
# checkpoint_path = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/Models/'+file_name
# print("loading ", file_name)
# mlp_classifier = load(checkpoint_path)

# '''
# Preict using SVM Classifier and test data
# '''
# # predict the classes on the images of the test set
# y_pred = mlp_classifier.predict(X_test_flattened)

# X_test, y_test, y_pred = shuffle(X_test, y_test, y_pred)
# X_test_img = X_test

# fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6), sharex=True, sharey=True)
# ax = axes.ravel()

# for i in range(4):
#     ax[i].imshow(X_test_img[i], cmap='gray')
#     ax[i].set_title(f'Label: {y_test[i]}\n Prediction: {y_pred[i]}')
#     ax[i].set_axis_off()
# fig.tight_layout
# plt.show()

# print(f"""Classification report for classifier {mlp_classifier}:\n
#       {metrics.classification_report(y_test, y_pred)}""")


'''
HOG
'''
# X_img,X_test, X_test_flattened,y_test = LoadTestDataHOG('all')
# X_img, X_test, X_test_flattened,y_test = LoadTestDataHOG(4)

# # SVM
# file_name = 'HOG-SVM.joblib'
# checkpoint_path = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/Models/'+file_name
# print("loading ", file_name)
# svm_classifier = load(checkpoint_path)

# '''
# Preict using SVM Classifier and test data
# '''
# # predict the classes on the images of the test set
# y_pred = svm_classifier.predict(X_test_flattened)

# X_img, y_test, y_pred = shuffle(X_img, y_test, y_pred)
# X_test_img = X_img

# fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6), sharex=True, sharey=True)
# ax = axes.ravel()

# for i in range(10):
#     ax[i].imshow(X_test_img[i], cmap='gray')
#     ax[i].set_title(f'Label: {y_test[i]}\n Prediction: {y_pred[i]}')
#     ax[i].set_axis_off()
# fig.tight_layout
# plt.show()

# print(f"""Classification report for classifier {svm_classifier}:\n
#       {metrics.classification_report(y_test, y_pred)}""")

# # MLP
# file_name = 'HOG-MLP.joblib'
# checkpoint_path = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/Models/'+file_name
# print("loading ", file_name)
# mlp_classifier = load(checkpoint_path)

# '''
# Preict using SVM Classifier and test data
# '''
# # predict the classes on the images of the test set
# y_pred = mlp_classifier.predict(X_test_flattened)

# X_img, y_test, y_pred = shuffle(X_img, y_test, y_pred)
# X_test_img = X_img

# fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6), sharex=True, sharey=True)
# ax = axes.ravel()

# for i in range(4):
#     ax[i].imshow(X_test_img[i], cmap='gray')
#     ax[i].set_title(f'Label: {y_test[i]}\n Prediction: {y_pred[i]}')
#     ax[i].set_axis_off()
# fig.tight_layout
# plt.show()

# print(f"""Classification report for classifier {mlp_classifier}:\n
#       {metrics.classification_report(y_test, y_pred)}""")



'''
SIFT
'''
# X_test, X_test_flattened,y_test = LoadTestDataSIFT('all')

# # MLP
# file_name = 'SIFT-MLP.joblib'
# checkpoint_path = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/Models/'+file_name
# print("loading ", file_name)
# mlp_classifier = load(checkpoint_path)

# '''
# Preict using SVM Classifier and test data
# '''
# # predict the classes on the images of the test set
# y_pred = mlp_classifier.predict(X_test_flattened)

# X_test, y_test, y_pred = shuffle(X_test, y_test, y_pred)
# X_test_img = X_test

# fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6), sharex=True, sharey=True)
# ax = axes.ravel()

# for i in range(4):
#     ax[i].imshow(X_test_img[i], cmap='gray')
#     ax[i].set_title(f'Label: {y_test[i]}\n Prediction: {y_pred[i]}')
#     ax[i].set_axis_off()
# fig.tight_layout
# plt.show()

# print(f"""Classification report for classifier {mlp_classifier}:\n
#       {metrics.classification_report(y_test, y_pred)}""")

