# Adding modules
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle

# Directory in which individual categories are located
DATADIR = "C:/Users/KILE/Desktop/"

categories = ["glass", "plastic", "cans"]

# Image width
IMG_W = int(640/2)
# Image height
IMG_H = int(480/2)

data_set = []

def creat_data_set():
    """Save images to data_set list"""
    # Iterate through the categories list
    for category in categories:  
        # A directory that contains images for a specific category
        path = os.path.join(DATADIR,category)  
        # Index of the current category in the categories list
        class_num = categories.index(category)  
        # Names of individual images
        for image in tqdm(os.listdir(path)):  
            try:
                # Image upload
                img_array = cv2.imread(os.path.join(path,image),cv2.IMREAD_GRAYSCALE) 
                # Image resizing
                new_array = cv2.resize(img_array, (IMG_W, IMG_H)) 
                # Adding an image and category index to the data_set list
                data_set.append([new_array, class_num]) 
            # Ignoring mistakes
            except Exception as e:
                pass
# Executing the creat_data_set() function
creat_data_set()
# Random order of the data_set list
random.shuffle(data_set)

# Create an empty list for X and Y
X = []
Y = []

# Iterate through the data_set list, and extract data from it
for features, label in data_set:
    # Adding features to X list.
    X.append(features)
    # Adding labels to Y list.
    Y.append(label)

# Saving X list to X.pickle
pickle_out = open("C:/Users/KILE/Desktop/X.pickle", "wb")
pickle.dump(X, pickle_out, protocol=4)
pickle_out.close()

# Saving Y list to Y.pickle
pickle_out = open("C:/Users/KILE/Desktop/Y.pickle", "wb")
pickle.dump(Y, pickle_out, protocol=4)
pickle_out.close()
