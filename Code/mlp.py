# import the necessary libraries
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load
from sklearn.model_selection import train_test_split

from base import LoadData
from base import LoadDataHOG

# X_train,X_train_flattened,y_train,X_test, X_test_flattened,y_test = LoadData(12271)
X_train,X_train_flattened,y_train,X_test, X_test_flattened,y_test = LoadData(100)
# X_train,X_train,X_train_flattened,y_train,X_test,X_test,X_test_flattened,y_test =LoadDataHOG(1000)

# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, stratify=y_train,random_state=1)

'''
create MLP classifier
'''

# Create a classifier: Multi-Layer Perceptron 
# classifier = MLPClassifier(hidden_layer_sizes=(50),activation='relu', max_iter=100, alpha=1e-4,
#                     solver='sgd', verbose=True, random_state=1,
#                     learning_rate_init=0.001)
# max 58%

classifier = MLPClassifier(
                    hidden_layer_sizes=(150),
                  #   activation='relu',
                    alpha=1e-4,
                    solver='sgd',
                    verbose=True,
                    learning_rate_init=0.001,
                    random_state=1
                    )
# max 59%

# classifier = MLPClassifier(
#                     hidden_layer_sizes=(50),
#                     alpha=1e-4,
#                     solver='sgd',
#                     # max_iter=100,
#                     verbose=True,
#                     learning_rate_init=0.001,
#                     random_state=1
#                     )

# adam

# classifier = MLPClassifier(hidden_layer_sizes=(50), activation='relu', alpha=1e-4,
#                     solver='sgd',verbose=True,
#                     learning_rate_init=)
                    
classifier.fit(X_train_flattened, y_train)

# file_name = 'MLP_MNIST.joblib'
# checkpoint_path = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/'+file_name

# dump(classifier, checkpoint_path) 

# classifier = load(checkpoint_path)



'''
Preict using SVM Classifier and test data
'''
# predict the classes on the images of the test set
y_pred = classifier.predict(X_test_flattened)





X_test, y_test, y_pred = shuffle(X_test, y_test, y_pred)
# X_test_img = X_test

# fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6), sharex=True, sharey=True)
# ax = axes.ravel()

# for i in range(10):
#     ax[i].imshow(X_test_img[i], cmap='gray')
#     ax[i].set_title(f'Label: {y_test[i]}\n Prediction: {y_pred[i]}')
#     ax[i].set_axis_off()
# fig.tight_layout
# plt.show()

# metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
# plt.show()

print(f"""Classification report for classifier {classifier}:\n
      {metrics.classification_report(y_test, y_pred)}""")






# https://stackoverflow.com/questions/49782584/early-stopping-and-regularization