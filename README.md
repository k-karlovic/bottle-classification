# Bottle classification
Classification of cans, plastic, and glass bottles using feature extraction and support vector machine method.
<br />
&nbsp;
## Table of Contents or Overview
* [Summary](#summary)
* [Setup](#setup)
* [Data set collection and manipulation](#data-set-collection-and-manipulation)
* [Algorithms for extracting image features](#algorithms-for-extracting-image-features)
* [Support vector machine](#support-vector-machine)
* [Conclusion](#conclusion)
* [Literature](#literature)

&nbsp;
## Summary
In this project, cans and bottles need to be classified using feature extraction. Classification is performed on cans, plastic, and glass bottles. The program is written in the Python programming language, and the main modules used for this project are OpenCV and scikitlearn. SIFT, SURF, and ORB algorithms are used to extract features on cans and bottles. The performance of ORB in feature detection is the same as with SIFT which means it is better than SURF and is faster by almost two orders of magnitude. The SVM algorithm is used to classify cans and bottles. The SVM algorithm tries to find the hyperplane with the largest possible margin between the support vectors. There are different kernels used by SVM algorithms, such as linear, RBF, and polynomial kernels.

&nbsp;
## Setup
### 1. Install Requirements
* [Python](https://www.python.org/downloads/)

To install the necessary packages in python run **`pip install -r requirements.txt`**.

### 2. 

### 3. 

### 4. 

### 5. 

### 6. 

### 7. 

&nbsp;
## Data set collection and manipulation
The data set includes three different categories of items. About 300 cans, plastic, and glass bottles were collected. The bottles were photographed with a webcam from the same height on a white paper that served as a substrate (Figure 1).

<p align="center"><i>
  Figure 1 Plastic packaging taken with a webcam
</i></p>

The cans and bottles need to be photographed from different angles because the camera cannot capture the whole bottle from the same position. This is achieved by having each bottle rotate around its axis and thus obtain images of the bottles from different angles. The aim is to automatically recognize the cans and bottles in any form it was in the environment, which is why plastic bottles are thermally processed and thus also receive a larger set of data (Figure 2).

<p align="center"><i>
  Figure 2 Heat-treated bottles
</i></p>


The bottles were re-photographed from all angles, with a cap, without a cap, with a label, and without a label. It is necessary to obtain a set of data with real situations of returning bottles where the bottles can be, for example, with a cap or without a label. A script was created in Python to photograph the bottles, which immediately converts the images to black and white (Figure 3).


<p align="center"><i>
  Figure 3 Black and white image of the bottle
</i></p>


&nbsp;
## Data set manipulation
To better train the model, it is necessary to increase the data set. This is achieved by reducing noise, rotating, and blurring images. OpenCV is a package in Python that contains all these options. Images are rotated 90 degrees 3 times with the `cv2.getRotationMatrix2D()` and `cv2.warpAffine()` functions. The `cv2.getRotationMatrix2D()` function must contain the center around which it will rotate, at what angle, and whether the image remains the same or shrinks or increases. The `cv2.warpAffine()` function combines the `cv2.getRotationMatrix2D()` command with the original image and rotates the images. The image of the rotated bottle is visible in Figure 4.

<p align="center"><i>
  Figure 4 Picture of a rotated bottle
</i></p>


The `cv2.fastNlMeansDenoising()` function is used to reduce noise. The function requires the original image and parameters for filter strength, templateWindowSize (recommended 7) and searchWindowSize (recommended 21). The result of reduced noise is visible in Figure 5.

<p align="center"><i>
  Figure 5 Figure with reduced noise
</i></p>


The function `cv2.medianBlur()` is used for blurring, which requires the original image and the parameter for blurring strength. Figure 6 shows the result of image blurring.

<p align="center"><i>
  Figure 6 Image blur
</i></p>


After rotating, reducing noise, and blurring the images, a data set of 496 cans, 144 glass, and 7543 plastic packaging was obtained. Image manipulation scripts are attached under `rotation.py`, `noise_reduction.py`, and `blurring.py`.

&nbsp;
## Algorithms for extracting image features
Finding features is an important task in many computer vision applications, such as finding images, discovering objects, and more.

The image feature extraction algorithms used in this project are:
* SIFT (Scale-Invariant Feature Transform),
* SURF (Speeded-Up Robust Features) and
* ORB (Oriented FAST and Rotated BRIEF).

### SIFT

The code below extracts the features of the images with the SIFT algorithm:

    # Adding modules
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from time import time`
    # Image upload
    img = cv2.imread('plasticna_boca.jpg')
    t0 = time()
    # Image upload
    sift = cv2.xfeatures2d.SIFT_create()
    # Identifying key points
    kp = sift.detect(img,None)
    # Prikazivanje brojakljucnih tocaka
    print(len(kp))
    #Prikazivanje vremena potrebno za izlucivanje znacajki SIFT algoritmom
    print("Vrijeme potrebno za izlucivanje znacajki SIFT algoritmom: %0.3fs" % time() - t0))
    # Crtanje ključnih točaka
    img=cv2.drawKeypoints(img,kp,img)
    # Spremanje slike
    cv2.imwrite('sift_plasticna_boca.jpg',img)
    #Prikazivanje slike
    plt.imshow(img)
    plt.show()

The `cv2.xfeatures2d.SIFT_create()` function is used to invoke / create the SIFT algorithm. The `sift.detect()` function finds the key point in the images. Each key point is a special structure that has many attributes such as its coordinates (x, y), the size of a significant neighborhood, the angle that determines its orientation, the response that indicates the strength of key points, etc. The `cv.drawKeyPoints()` function draws small circles at key points. If the `cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS` function is added, it will draw a circle with the size of the key point and show the orientation. The SIFT algorithm found 94 key points which are shown in Figure 7.

<p align="center"><i>
  Figure 7 Extraction of features by SIFT algorithm
</i></p>


### SURF

The code below extracts the features by the SURF algorithm.

    # Dodavanje modula
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from time import time
    # Ucitavanje slike
    img = cv2.imread('plasticna_boca.jpg')
    t0 = time()
    # Kreiranje metode SURF
    surf = cv2.xfeatures2d.SURF_create()
    # Prepoznavanje ključnih točaka
    kp = surf.detect(img,None)
    # Prikazivanje brojakljucnih tocaka
    print(len(kp))
    #Prikazivanje vremena potrebno za izlucivanje znacajki SURF algoritmom
    print("Vrijeme potrebno za izlucivanje znacajki SURF algoritmom: %0.3fs" % (time() - t0))
    # Crtanje ključnih točaka
    img=cv2.drawKeypoints(img,kp,img)
    # Spremanje slike
    cv2.imwrite('surf_plasticna_boca.jpg',img)
    #Prikazivanje slike
    plt.imshow(img)
    plt.show()

The `cv2.xfeatures2d.SURF_create()` function is used to invoke / create the SURF algorithm. The `surf.detect()` function finds a key point in the images. Each key point is a special structure that has many attributes such as its coordinates (x, y), the size of a significant neighborhood, the angle that determines its orientation, the response that indicates the strength of key points, etc. Function `cv.drawKeyPoints()` draws small circles at the locations of key points. If the `cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS` function is added, it will draw a circle with the size of the key point and show the orientation. The SURF algorithm found 162 key points shown in Figure 8.


<p align="center"><i>
  Figure 8 Extraction of features by SURF algorithm
</i></p>


### ORB
The code below extracts features from the ORB algorithm

    # Dodavanje modula
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from time import time
    # Ucitavanje slike
    img = cv2.imread('plasticna_boca.jpg')
    t0 = time()
    # Kreiranje metode ORB
    orb = cv2.ORB_create()
    # Prepoznavanje ključnih točaka
    kp = orb.detect(img,None)
    # Prikazivanje brojakljucnih tocaka
    print(len(kp))
    23
    #Prikazivanje vremena potrebno za izlucivanje znacajki ORB algoritmom
    print("Vrijeme potrebno za izlucivanje znacajki ORB algoritmom: %0.3fs" % (time() - t0))
    # Crtanje ključnih točaka
    img=cv2.drawKeypoints(img,kp,img)
    # Spremanje slike
    cv2.imwrite('orb_plasticna_boca.jpg',img)
    #Prikazivanje slike
    plt.imshow(img)
    plt.show()

The `cv2.ORB_create()` function is used to invoke/create the ORB algorithm. The `orb.detect()` function finds a key point in the images. Each key point is a special structure that has many attributes such as its coordinates (x, y), the size of a significant neighborhood, the angle that determines its orientation, the response that indicates the strength of key points, etc. Function `cv.drawKeyPoints()` draws small circles at the locations of key points. The ORB algorithm found 467 key points shown in Figure 9.

<p align="center"><i>
  Figure 9 Extraction of features by ORB algorithm
</i></p>


### Comparison of obtained results

Figure 10 shows a comparison of feature extraction for SIFT, SURF, and ORB algorithms.

<p align="center"><i>
  Figure 10 Comparison of the obtained results for SIFT, SURF, and ORB
</i></p>



Figure 10 shows that when extracting features from the ORB algorithm, all key points are on the plastic packaging. This means that he best recognized the key points because it is a subject that is classified using the SVM algorithm and also gives better results when testing the SVM algorithm. Table 1 shows the number of features found on plastic packaging and the time required to extract features by SIFT, SURF, and ORB algorithms.

<p align="center"><i>
  Table 1 Number of features and time required for feature extraction
</i></p>

The ORB algorithm is faster than the SIFT algorithm, but because it has found more features it also takes more time.

&nbsp;
## Support vector machine

&nbsp;
## Conclusion

&nbsp;
## Literature

