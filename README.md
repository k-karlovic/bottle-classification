# Bottle classification
Classification of cans, plastic, and glass bottles using feature extraction and support vector machine method.
<br />
&nbsp;
## Table of Contents
* [Summary](#summary)
* [Setup](#setup)
* [Data set collection](#data-set-collection)
* [Data set manipulation](#data-set-manipulation)
* [Algorithms for extracting image features](#algorithms-for-extracting-image-features)
* [Support vector machine](#support-vector-machine)
* [Cans and bottles classification model](#cans-and-bottles-classification-model)
* [Conclusion](#conclusion)
* [Literature](#literature)

&nbsp;
## Summary
In this project, cans and bottles need to be classified using feature extraction. Classification is performed on cans, plastic, and glass bottles. The program is written in the Python programming language, and the main modules used for this project are OpenCV and scikitlearn. SIFT, SURF, and ORB algorithms are used to extract features on cans and bottles. The performance of ORB in feature detection is the same as with SIFT which means it is better than SURF and is faster by almost two orders of magnitude. The SVM algorithm is used to classify cans and bottles. The SVM algorithm tries to find the hyperplane with the largest possible margin between the support vectors. There are different kernels used by SVM algorithms, such as linear, RBF, and polynomial kernels.

&nbsp;
## Setup
### 1. Clone the repository
  - Open Git Bash 
  - Change the directory to the location wehere you want clone the directory
  - Type git clone https://github.com/k-karlovic/bottle-classification.git

<br />

### 2. Download the data set
  - Go to [https://www.dropbox.com/s/0luqxkj0axpgsxo/bottles.zip?dl=0](https://www.dropbox.com/s/0luqxkj0axpgsxo/bottles.zip?dl=0)
  - Dwnload the data set
  - Unzip the file

<br />

### 3.  Install Requirements
* [Python](https://www.python.org/downloads/)

To install the necessary packages in python run **`pip install -r requirements.txt`**.

<br />

### 4. Run the `data_set_preparation.py` script
Run the `data_set_preparation.py` script to save the data set in pickle format.

<br />

### 5. Run the `svmn.py` script
Run the `svmn.py` script to train and test the SVM model.

&nbsp;
## Data set collection
The data set includes three different categories of items. About 300 cans, plastic, and glass bottles were collected. The bottles were photographed with a webcam from the same height on a white paper that served as a substrate (Figure 1).

<br />

<p align="center">
  <img width="45%" src="https://github.com/k-karlovic/bottle-classification/blob/main/images/plastic_bottle_taken_with_a_webcam.jpg?raw=true"/>
</p>
<p align="center"><i>
  Figure 1 Plastic bottle taken with a webcam
</i></p>

<br />

The cans and bottles need to be photographed from different angles because the camera cannot capture the whole bottle from the same position. This is achieved by having each bottle rotate around its axis and thus obtain images of the bottles from different angles. The aim is to automatically recognize the cans and bottles in any form it was in the environment, which is why plastic bottles are thermally processed and thus also receive a larger set of data (Figure 2).

<br />

<p align="center">
  <img width="45%" src="https://github.com/k-karlovic/bottle-classification/blob/main/images/heat-treated_bottles.jpg?raw=true"/>
</p>
<p align="center"><i>
  Figure 2 Heat-treated bottles
</i></p>

<br />

The bottles were re-photographed from all angles, with a cap, without a cap, with a label, and without a label. It is necessary to obtain a set of data with real situations of returning bottles where the bottles can be, for example, with a cap or without a label. A script was created in Python to photograph the bottles, which immediately converts the images to black and white (Figure 3).

<br />

<p align="center">
  <img width="60%" src="https://github.com/k-karlovic/bottle-classification/blob/main/images/black_and_white_image_of_the_bottle.jpg?raw=true"/>
</p>
<p align="center"><i>
  Figure 3 Black and white image of the bottle
</i></p>

<br />

&nbsp;
## Data set manipulation
To better train the model, it is necessary to increase the data set. This is achieved by reducing noise, rotating, and blurring images. OpenCV is a package in Python that contains all these options. Images are rotated 90 degrees 3 times with the `cv2.getRotationMatrix2D()` and `cv2.warpAffine()` functions. The `cv2.getRotationMatrix2D()` function must contain the center around which it will rotate, at what angle, and whether the image remains the same or shrinks or increases. The `cv2.warpAffine()` function combines the `cv2.getRotationMatrix2D()` command with the original image and rotates the images. The image of the rotated bottle is visible in Figure 4.

<br />

<p align="center">
  <img width="60%" src="https://github.com/k-karlovic/bottle-classification/blob/main/images/rotated_bottle.jpg?raw=true"/>
</p>
<p align="center"><i>
  Figure 4 Rotated bottle
</i></p>

<br />

The `cv2.fastNlMeansDenoising()` function is used to reduce noise. The function requires the original image and parameters for filter strength, templateWindowSize (recommended 7) and searchWindowSize (recommended 21). The result of reduced noise is visible in Figure 5.

<br />

<p align="center">
  <img width="60%" src="https://github.com/k-karlovic/bottle-classification/blob/main/images/reduced_noise.jpg?raw=true"/>
</p>
<p align="center"><i>
  Figure 5 Reduced noise
</i></p>

<br />

The function `cv2.medianBlur()` is used for blurring, which requires the original image and the parameter for blurring strength. Figure 6 shows the result of image blurring.

<br />

<p align="center">
  <img width="60%" src="https://github.com/k-karlovic/bottle-classification/blob/main/images/blurring.jpg?raw=true"/>
</p>
<p align="center"><i>
  Figure 6 Blurring
</i></p>

<br />

After rotating, reducing noise, and blurring the images, a data set of 496 cans, 144 glass, and 7543 plastic bottles were obtained. Image manipulation scripts are attached under `rotation.py`, `noise_reduction.py`, and `blurring.py`.

&nbsp;
## Algorithms for extracting image features
Finding features is an important task in many computer vision applications, such as finding images, discovering objects, and more.

The image feature extraction algorithms used in this project are:
* SIFT (Scale-Invariant Feature Transform),
* SURF (Speeded-Up Robust Features) and
* ORB (Oriented FAST and Rotated BRIEF).

<br />

### SIFT

The code below extracts the features from the images with the SIFT algorithm:

<br />

    # Adding modules
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from time import time
    # Image upload
    img = cv2.imread('plastic_bottle.jpg')
    t0 = time()
    # Creating SIFT method
    sift = cv2.xfeatures2d.SIFT_create()
    # Identifying key points
    kp = sift.detect(img,None)
    # Displaying the number of key points
    print(len(kp))
    # Displaying the time required to extract features using the SIFT algorithm
    print("Time required to extract features by SIFT algorithm: %0.3fs" % time() - t0))
    # Drawing key points
    img=cv2.drawKeypoints(img,kp,img)
    # Saving the image
    cv2.imwrite('sift_plastic_bottle.jpg',img)
    # Image display
    plt.imshow(img)
    plt.show()

<br />

The `cv2.xfeatures2d.SIFT_create()` function is used to invoke / create the SIFT algorithm. The `sift.detect()` function finds key points in images. Each key point is a special structure that has many attributes such as its coordinates (x, y), the size of a significant neighborhood, the angle that determines its orientation, the response that indicates the strength of key points, etc. The `cv.drawKeyPoints()` function draws small circles at key points. If the `cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS` function is added, it will draw a circle with the size of the key point and show the orientation. The SIFT algorithm found 94 key points which are shown in Figure 7.

<br />

<p align="center">
  <img width="70%" src="https://github.com/k-karlovic/bottle-classification/blob/main/images/extraction_of_features_by_SIFT_algorithm.jpg?raw=true"/>
</p>
<p align="center"><i>
  Figure 7 Extraction of features by SIFT algorithm
</i></p>

<br />

### SURF

The code below extracts the features of the images with the SURF algorithm:

<br />

    # Adding modules
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from time import time
    # Image upload
    img = cv2.imread('plastic_bottle.jpg')
    t0 = time()
    # Creating SURF method
    surf = cv2.xfeatures2d.SURF_create()
    # Identifying key points
    kp = surf.detect(img,None)
    # Displaying the number of key points
    print(len(kp))
    # Displaying the time required to extract features using the SURF algorithm
    print("Time required to extract features by SURF algorithm: %0.3fs" % (time() - t0))
    # Drawing key points
    img=cv2.drawKeypoints(img,kp,img)
    # Saving the image
    cv2.imwrite('surf_plastic_bottle.jpg',img)
    # Image display
    plt.imshow(img)
    plt.show()

<br />

The `cv2.xfeatures2d.SURF_create()` function is used to invoke / create the SURF algorithm. The `surf.detect()` function finds key points in images. Each key point is a special structure that has many attributes such as its coordinates (x, y), the size of a significant neighborhood, the angle that determines its orientation, the response that indicates the strength of key points, etc. Function `cv.drawKeyPoints()` draws small circles at the locations of key points. If the `cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS` function is added, it will draw a circle with the size of the key point and show the orientation. The SURF algorithm found 162 key points shown in Figure 8.

<br />

<p align="center">
  <img width="70%" src="https://github.com/k-karlovic/bottle-classification/blob/main/images/extraction_of_features_by_SURF_algorithm.jpg?raw=true"/>
</p>
<p align="center"><i>
  Figure 8 Extraction of features by SURF algorithm
</i></p>

<br />

### ORB
The code below extracts the features of the images with the ORB algorithm:

<br />

    # Adding modules
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from time import time
    # Image upload
    img = cv2.imread('plastic_bottle.jpg')
    t0 = time()
    # Creating ORB method
    orb = cv2.ORB_create()
    # Identifying key points
    kp = orb.detect(img,None)
    # Displaying the number of key points
    print(len(kp))
    # Displaying the time required to extract features using the ORB algorithm
    print("Time required to extract features by ORB algorithm: %0.3fs" % (time() - t0))
    # Drawing key points
    img=cv2.drawKeypoints(img,kp,img)
    # Saving the image
    cv2.imwrite('orb_plastic_bottle.jpg',img)
    # Image display
    plt.imshow(img)
    plt.show()

<br />

The `cv2.ORB_create()` function is used to invoke/create the ORB algorithm. The `orb.detect()` function finds key points in images. Each key point is a special structure that has many attributes such as its coordinates (x, y), the size of a significant neighborhood, the angle that determines its orientation, the response that indicates the strength of key points, etc. Function `cv.drawKeyPoints()` draws small circles at the locations of key points. The ORB algorithm found 467 key points shown in Figure 9.

<br />
<br />

<p align="center">
  <img width="70%" src="https://github.com/k-karlovic/bottle-classification/blob/main/images/extraction_of_features_by_ORB_algorithm.jpg?raw=true"/>
</p>
<p align="center"><i>
  Figure 9 Extraction of features by ORB algorithm
</i></p>

<br />

### Comparison of obtained results

Figure 10 shows a comparison of feature extraction for SIFT, SURF, and ORB algorithms.

<br />

<p align="center">
  <img width="95%" src="https://github.com/k-karlovic/bottle-classification/blob/main/images/comparison_of_the_obtained_results_SIFT_SURF_ORB.jpg?raw=true"/>
</p>
<p align="center"><i>
  Figure 10 Comparison of the obtained results for SIFT, SURF, and ORB
</i></p>

<br />

Figure 10 shows that when extracting features from the ORB algorithm, all key points are on the plastic bottle. This means that he best recognized the key points because it is a subject that is classified using the SVM algorithm and also gives better results when testing the SVM algorithm. 
Table 1 shows the number of features found on the plastic bottle and the time required to extract features by SIFT, SURF, and ORB algorithms.

<br />

<p align="center"><i>
  Table 1 Number of features and time required for feature extraction
</i></p>

<div align="center">
<table>
    <thead>
        <tr>
            <th align="center">Feature extraction algorithms</th>
            <th align="center">Number of key points</th>
            <th align="center">Algorithm time [s]</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th align="center">SIFT</th>
            <th align="center">94</th>
            <th align="center">90.105</th>
        </tr>
        <tr>
            <th align="center">SURF</th>
            <th align="center">162</th>
            <th align="center">0.256</th>
        </tr>
        <tr>
            <th align="center">ORB</th>
            <th align="center">467</th>
            <th align="center">0.241</th>
        </tr>
    </tbody>
</table>
</div>

<br />

The ORB algorithm is faster than the SIFT algorithm, but because it has found more features it also takes more time.

&nbsp;
## Support vector machine
Support Vector Machines (SVMs) are supervised machine learning algorithms that are used for classification and regression purposes (Figure 11).

<br />

<p align="center">
  <img width="70%" src="https://github.com/k-karlovic/bottle-classification/blob/main/images/svm.png?raw=true"/>
</p>
<p align="center"><i>
  Figure 11 Support Vector Machine
</i></p>

<br />

#### Hyperplane

A hyperplane is a decision boundary which separates between given set of data points having different class labels. The SVM classifier separates data points using a hyperplane with the maximum amount of margin.

#### Support vectors

Support vectors are the sample data points, which are closest to the hyperplane.

#### Margin

A margin is a separation gap between the two lines on the closest data points. It is calculated as the perpendicular distance from the line to support vectors or closest data points.

In SVMs, our main objective is to select a hyperplane with the maximum possible margin between support vectors in the given dataset. SVM searches for the maximum margin hyperplane in the following 2 step process:

* Generate hyperplanes which segregates the classes in the best possible way. There are many hyperplanes that might classify the data. We should look for the best hyperplane that represents the largest separation, or margin, between the two classes.
* So, we choose the hyperplane so that distance from it to the support vectors on each side is maximized. If such a hyperplane exists, it is known as the maximum margin hyperplane and the linear classifier it defines is known as a maximum margin classifier.

#### Kernel
A kernel transforms a low-dimensional input data space into a higher dimensional space. So, it converts non-linear separable problems to linear separable problems by adding more dimensions to it.

Some types of kernels:

* Linear kernel
* Polynomial kernel
* Radial Basis Function (RBF) kernel


&nbsp;
## Cans and bottles classification model
The system for the classification of cans and bottles using the extraction of features and algorithms of artificial intelligence consists of:
* preparation of a data set,
* feature extraction and
* training and testing of features by SVM algorithm.

<br />

### Preparation of a data set

The collected data set (Chapter “2. Data collection and manipulation) needs to be prepared for the extraction algorithm. Each image is loaded with the "cv2.imread ()" function and reduced with the "cv2.resize ()" function. 320x240 images are saved in sheet X and the class index (0 for glass bottles, 1 for plastic bottles and 2 for cans) is saved in sheet Y. Lists X and Y are saved using the "pickle.dump ()" function in X files .pickle and Y.pickle. The code for preparing the data set is attached as "Preparation of DataSet.py".

The following code displays the images as input data in the form of dots, as shown in Figure 12.

<br />

    # Adding modules
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle
    from sklearn import preprocessing
    # Loading X and Y
    pickle_in = open("X.pickle", "rb")
    X = pickle.load(pickle_in)
    pickle_in = open("Y.pickle", "rb")
    Y = pickle.load(pickle_in)
    # Image width
    IMG_W = int(640/2)
    # Image height
    IMG_H = int(480/2)
    # Creating a vector of shape len(labels), len(list_data[0][0])
    X = np.array(X).reshape(len(X),IMG_W*IMG_H)
    Y = np.array(Y)
    # Categories
    Categories = ["Glass", "Plastic", "Cans"]
    # The color of the points of individual categories
    colors = ['red', 'green', 'blue']
    # Draw points on a graph
    for i in range(len(colors)):
    xs = X[:, 1][Y == i]
    ys = X[:, 2][Y == i]
    plt.scatter(xs, ys, c=colors[i], alpha=0.5)
    # Legend
    plt.legend(Categories)
    # Saving the graph
    plt.savefig('input_datai.jpg')
    plt.show()

<br />


<p align="center">
  <img width="70%" src="https://github.com/k-karlovic/bottle-classification/blob/main/images/input_data.jpg?raw=true"/>
</p>
<p align="center"><i>
  Figure 12 Input data
</i></p>

<br />

### Extraction features

The project uses 3 algorithms for extracting features, as in Chapter “4. Algorithms for extracting image features " shown:
* SIFT,
* SURF i
* ORB.

<br />

#### SIFT

The code below was used to extract the features of all the images in the data set using the SIFT algorithm.

<br />

    def sift(img):
      """Create SIFT method to exclude features, and return kp and des"""
      # Creating SIFT method
      sift = cv2.xfeatures2d.SIFT_create()
      # Determining the number of features and features
      kp, des = sift.detectAndCompute(img,None)
      return kp, des

<br />

#### SURF

The code below was used to extract the features of all the images in the data set using the SURF algorithm.

<br />

    def surf(img):
      """Create SURF method to exclude features, and return kp and des"""
        # Create SURF method
      surf = cv2.xfeatures2d.SURF_create()
      # Determining the number of features and features
      kp, des = surf.detectAndCompute(img,None)
      return kp, des

<br />

#### ORB

The code below was used to extract the features of all the images in the data set using the ORB algorithm.

<br />

    def orb(img):
      """Create ORB method to exclude features, and return kp and des"""
      # Create ORB method
      orb = cv2.ORB_create()
      # Determining the number of features and features
      kp, des = orb.detectAndCompute(img,None)
      return kp, des

<br />
<br />

The input must be the same size to train the SVM model, which means that the number of features for each image should be the same. This will be achieved by the following code:

<br />

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

<br />

The code only accepts images that have more than 20 features.

<br />

    if len(kp) < 20:
      continue

<br />

Only the first 20 features that are added to the list are taken.

<br />

    des = des[0:20,:]

<br />

Before training and testing features with the SVM algorithm, it is necessary to divide the features into trained and tested data. This means that 70% of the images will be a training set and 30% a test set.

<br />

    # Division of dataset into trained and tested data
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

<br />

### Training and testing of features by SVM algorithm

For training, it is necessary to determine parameters, such as the core and parameters C and γ. There are two variants:
* self-testing with different parameters or
* use the "GridSearchCV()" function - automatic search.

<br />

#### GridSearchCV()

The following code shows the procedure for using the "GridSearchCV ()" function, which throws out clf.best_estimator_ as a result of the best parameters.

<br />

    def svm_parameters(X_train, y_train):
      """Finding parameters for model training and returning clf.best_estimator_"""
      t0 = time()
      # Parameters
      param_grid = {'C': [1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.001, 0.01, 0.1], 
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
      # Parameter search function
      clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid)
      clf = clf.fit(X_train, y_train)
      print("Parameter finding time: %0.3fs" % (time() - t0))
      return clf.best_estimator_

<br />

It is defined for which training and testing algorithm "svm.SVC()" is used, and parameter ranges are defined.

<br />

    # Parameters
      param_grid = {'C': [1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.001, 0.01, 0.1], 
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

<br />

The time to find the best parameters took more than 5 days. The results obtained are:

<br />

    clf = svm.SVC(C=1000, cache_size=200, class_weight='balanced', coef0=0.0, decision_function_shape='ovr', 
                  degree=3, gamma=1e-4, kernel='rbf', max_iter=-1, probability=False, random_state=None, 
                  shrinking=True, tol=0.001, verbose=False)

<br />

Parameter C did not reach the minimum number given, but gamma did, which means that it is necessary to check whether the gamma less than 0.0001 gives better results. After checking, it was obtained that gamma = 1e-8 gives the best result:

<br />

    clf = svm.SVC(C=1000, cache_size=200, class_weight='balanced', coef0=0.0, decision_function_shape='ovr', 
                  degree=3, gamma=1e-8, kernel='rbf', max_iter=-1, probability=False, random_state=None, 
                  shrinking=True, tol=0.001, verbose=False)

<br />

#### Testing

The accuracy of the trained model is always higher than the tested one because during testing the model tests with new input data. The clf.predict (X_test) command is used for testing to obtain the model accuracy percentage. The result can be displayed through the accuracy or precision of the model.

&nbsp;
## Conclusion
The project consists of collecting images to create a data set for training the SVM algorithm. The images were collected by photographing bottles and cans. The bottles were photographed with a cap, without a cap, with a label, and without a label in order to get as realistic a situation as possible. Plastic bottles were also subsequently heat-treated and photographed. The OpenCV image processing package in Python was used to increase the data set. In order to reduce the time required to train the model, feature extraction algorithms are used. There are different algorithms for extracting features, and SIFT, SURF, and ORB algorithms were used in the project. The performance of ORB in feature detection is the same as with SIFT which means it is better than SURF and is faster by almost two orders of magnitude. In Chapter “7. Comparison of the obtained results "using the function" GridSearchCV () "the optimal parameters are obtained. The RBF and polynomial cores with the SURF algorithm for finding features have the highest precision, the lowest training and testing times of SVM models. But the ORB algorithm has the least feature finding time, which also results in a faster classification of cans and bottles. Therefore, the best solution is offered by the ORB algorithm, which has an accuracy of 0.927 and 7 times shorter time compared to the SURF algorithm. The application of feature extraction algorithms directly results in a reduction in the time required to train the model.

&nbsp;
## Literature
"Geometric Image Transformations“, https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html

„Image Denoising“, https://opencv-pythontutroals.readthedocs.io/en/latest/py_tutorials/py_photo/py_non_local_means/py_non_local_means.html

„Image Filtering“, https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html

„Scale-invariant feature transform“, https://en.wikipedia.org/wiki/Scaleinvariant_feature_transform

„Introduction to SIFT( Scale Invariant Feature Transform)“, https://medium.com/data-breach/introduction-to-sift-scale-invariant-feature-transform-65d7f3a72d40

„Introduction to SIFT (Scale-Invariant Feature Transform)“, https://opencvpythontutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html

„Introduction to SURF (Speeded-Up Robust Features)“, https://medium.com/data-breach/introduction-to-surf-speeded-up-robust-featuresc7396d6e7c4e

„Introduction to SURF (Speeded-Up Robust Features)“, https://opencvpythontutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html

„Introduction to ORB (Oriented FAST and Rotated BRIEF)“, https://medium.com/data-breach/introduction-to-orb-oriented-fast-and-rotated-brief-4220e8ec40cf

„Support Vector Machines Classifier Tutorial with Python“, https://www.kaggle.com/prashant111/svm-classifier-tutorial

Garreta R., Moncecchi G.: "Learning scikit-learn: Machine Learning in Python", Packt Publishing, UK, 2013.

Raschka S.: "Python Machine Learning", Packt Publishing, UK, 2015.
