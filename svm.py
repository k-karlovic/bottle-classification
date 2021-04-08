# Adding modules
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.externals import joblib

def data(X_path,Y_path):
  """Open .pickle files, and restore X and Y lists"""
  # Open the X.pickle file
  pickle_in = open(X_path, "rb")
  X = pickle.load(pickle_in)
  # Open the Y.pickle file
  pickle_in = open(Y_path, "rb")
  Y = pickle.load(pickle_in)
  return X, Y

def sift(img):
  """Create SIFT method to exclude features, and return kp and des"""
  # Creating SIFT method
  sift = cv2.xfeatures2d.SIFT_create()
  # Determining the number of features and features
  kp, des = sift.detectAndCompute(img,None)
  return kp, des

def orb(img):
  """Create ORB method to exclude features, and return kp and des"""
  # Create ORB method
  orb = cv2.ORB_create()
  # Determining the number of features and features
  kp, des = orb.detectAndCompute(img,None)
  return kp, des

def surf(img):
  """Create SURF method to exclude features, and return kp and des"""
    # Create SURF method
  surf = cv2.xfeatures2d.SURF_create()
  # Determining the number of features and features
  kp, des = surf.detectAndCompute(img,None)
  return kp, des

def feature_number(feature):
  """Creating a list with the features of individual images, and returning list_data and ind"""
  # Creating a blank list ind
  ind = []
  # Create a blank list_data list
  list_data = []
  t0 = time()
  # Iteration from 0 to the total number of data in X
  for i in range(len(X)):
    # Execution of SIFT, SURF and ORB functions
    kp, des = feature(X[i])
    # If the number of features in that image is less than 20, the image does not qualify
    if len(kp) < 20:
      # Adding to ind list
      ind.append(i)
      continue
    # Forming a feature of equal size (equal number of data)
    des = des[0:20,:]
    # Formation of the obtained feature data in the form 1, len (des) * len (des [1])
    vector_data = des.reshape(1,len(des)*len(des[1]))
    # Adding vector_data to the list_data list
    list_data.append(vector_data)
  # List of names of feature extraction methods
  features = ['sift', 'surf', 'orb']
  print("Algorithm time: %0.3fs" % (time() - t0))
  return list_data, ind
    
def svm_parameters(X_train, y_train):
  """Finding parameters for model training and returning clf.best_estimator_"""
  t0 = time()
  # Parameters
  param_grid = {'C': [1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.001, 0.01, 0.1], 
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
  # Parameter search function
  clf = GridSearchCV(
    svm.SVC(kernel='rbf', class_weight='balanced'), param_grid)
  clf = clf.fit(X_train, y_train)
  print("Parameter finding time: %0.3fs" % (time() - t0))
  return clf.best_estimator_

def svm_train(X_train, y_train):
  """Model training and returning clf"""
  t0 = time()
  # Creating an SVM classifier
  clf = svm.SVC(C=1000, cache_size=200, class_weight='balanced', coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=1e-8, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
  # Training of SVM classification model
  clf.fit(X_train, y_train)
  print("Model training time: %0.3fs" % (time() - t0))
  return clf

def svm_test(clf, X_test, y_test):
  """Testing the model and returning y_pred"""
  t0 = time()
  # Testing of SVM classification model
  y_pred = clf.predict(X_test)
  # Model accuracy: what percentage is accurately classified data? (TP + TN) / (TP + TN + FP + FN)
  print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
  # Model precision: what is the percentage of positive identifications in a set of positively classified data? TP / (TP + FP)
  print("Precision:",metrics.precision_score(y_test, y_pred, average='micro'))
  # Model recall: what is the percentage of positive identifications in the set of all positive data? TP / (TP + FN)
  print("Recall:",metrics.recall_score(y_test, y_pred, average='micro'))
  # Table of results obtained
  print(classification_report(y_test, y_pred, target_names=categories))
  print("Model testing time: %0.3fs" % (time() - t0))
  return y_pred

def svm_save(clf, path):
  """Saving SVM model"""
  joblib.dump(clf, path)

def plot_gallery(images, titles, h, w, n_row=1, n_col=2):
  """Displays individual images, image categories, and default categories"""
  # Image window size
  plt.figure(figsize=(4 * n_col, 2 * n_row))
  # Image parameters
  plt.subplots_adjust(bottom=0, left=0.1, right=0.9, top=.95, hspace=.35)
  # Display a certain number of images
  for i in range(n_row * n_col):
    plt.subplot(n_row, n_col, i + 1)
    plt.imshow(images[i].reshape((w,h)))
    plt.title(titles[i], size=10)
    plt.xticks(())
    plt.yticks(())

def title(y_pred, y_test, target_names, i):
  """Extract the actual and default image categories, and return pred_name and true_name"""
  pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
  true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
  return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

if __name__ == '__main__': 
  # List of categories
  categories = ["Glass", "Plastic", "Cans"]
  # The directory where the X and Y data is located
  X_path = "C:\\Users\\KILE\\Desktop\\X.pickle"
  Y_path = "C:\\Users\\KILE\\Desktop\\Y.pickle"
  # Image width
  IMG_W = int(640/2)
  # Image height
  IMG_H = int(480/2)
  # Executing the data() function
  X, Y = data(X_path, Y_path)
  # List of names of feature extraction methods
  features = ['sift', 'surf', 'orb']
  a = 0
  # Iterate through individual features
  for feature in [sift, surf, orb]:
    t1 = time()
    # Copying data from Y
    labels = Y[:]
    # Executing the feature_number() function
    list_data, ind = feature_number(feature)
    # Iterate through the list to delete data that didn't meet a sufficient number of features.
    for i in sorted(ind, reverse=True):
      del labels[i]
    # Creating a vector in the form of len (labels), len (list_data [0] [0])
    data = np.array(list_data).reshape(len(labels),len(list_data[0][0]))
    # Creating a vector
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    # Division of dataset into trained and tested data
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3,random_state=42) # 70% training and 30% test
    # Executing the svm_parameters() function
    #clf = svm_parameters(X_train, y_train)
    # Executing the svm_train() function
    clf = svm_train(X_train, y_train)
    # Executing the svm_test() function
    y_pred = svm_test(clf, X_test, y_test)
    print("The time of the whole program: %0.3fs" % (time() - t1))
    # The directory where the model is stored
    save_path = "C:\\Users\\KILE\\Desktop\\" + str(features[a]) + "_trained_model.npy"
    # Executing the svm_save() function
    svm_save(clf, save_path)
    a += 1
    # Execution of the title() function where "i" is the number of images tested
    prediction_titles = [title(y_pred, y_test, categories, i)
                        for i in range(y_pred.shape[0])]
    # Execution of the plot_gallery () function for SIFT
    #plot_gallery(X_test, prediction_titles, 80,80)
    # Execution of the plot_gallery () function for SURF
    #plot_gallery(X_test, prediction_titles, 50,64)
    # Execution of the plot_gallery () function for ORB
    #plot_gallery(X_test, prediction_titles, 80,20)
    plt.show()
  
