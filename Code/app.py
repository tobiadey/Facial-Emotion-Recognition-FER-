'''
Load data
'''
# from base import *
from base2 import *


'''
load data with no feature dectection, just tis regual pixel values
'''
# X_train,y_train,X_test,y_test = LoadData(100)
# X_train,X_train_flattened,y_train,X_test, X_test_flattened,y_test = LoadData(100)
# print("X_train ",X_train.shape)
# print("y_train ", y_train.shape)
# print("---------------------------")
# print("X_test ",X_test.shape)
# print("y_test ", y_test.shape)

# fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5), sharex=True, sharey=True)
# ax = axes.ravel()

# for i in range(10):
#     ax[i].imshow(X_train[i], cmap='gray')
#     ax[i].set_title(f'Label: {y_train[i]}')
#     ax[i].set_axis_off()
# fig.tight_layout()
# plt.show()

'''
load data with with HOG feature dectection
'''

# X_train,X_train_HOG,X_train_HOG_flattened,y_train,X_test,X_test_HOG,X_test_HOG_flattened,y_test =LoadDataHOG(100)

# print("X_train ",X_train.shape)
# print("X_train_HOG ",X_train_HOG.shape)
# print("X_train_HOG_flattened ",X_train_HOG_flattened.shape)
# print("y_train ", y_train.shape)
# print("---------------------------")
# print("X_test ",X_test.shape)
# print("X_test_HOG ",X_test_HOG.shape)
# print("X_test_HOG_flattened ",X_test_HOG_flattened.shape)
# print("y_test ", y_test.shape)
# fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5), sharex=True, sharey=True)
# ax = axes.ravel()

# for i in range(10):
#     ax[i].imshow(X_train_HOG[i], cmap='gray')
#     ax[i].set_title(f'Label: {y_train[i]}')
#     ax[i].set_axis_off()
# fig.tight_layout()
# plt.show()

'''
load data with with SIFT feature dectection
'''

X_train_SIFT,X_train_SIFT_flattened,y_train,X_test_SIFT,X_test_SIFT_flattened,y_test =LoadDataSIFT(100)

# print("X_train ",X_train.shape)
print("X_train_SIFT ",X_train_SIFT.shape)
print("X_train_SIFT_flattened ",X_train_SIFT_flattened.shape)
print("y_train ", y_train.shape)
print("---------------------------")
# print("X_test ",X_test.shape)
print("X_test_SIFT ",X_test_SIFT.shape)
print("X_test_SIFT_flattened ",X_test_SIFT_flattened.shape)
print("y_test ", y_test.shape)
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5), sharex=True, sharey=True)
ax = axes.ravel()

for i in range(10):
    ax[i].imshow(X_train_SIFT[i], cmap='gray')
    ax[i].set_title(f'Label: {y_train[i]}')
    ax[i].set_axis_off()
fig.tight_layout()
plt.show()


# test()
