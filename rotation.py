# Adding modules
import cv2
import os 
import random
import string
  
def rot(location): 
    """rotating individual images""" 
    # Iterating through individual images in a defined location
    for imagename in os.listdir(location):
        # Creating a source from the location and image name
        src = location + imagename
        # Image upload 
        image = cv2.imread(src)
        # Getting image size
        (h, w) = image.shape[:2]
        # Getting the center of the image
        center = (w / 2, h / 2)
        # Defining rotation angles
        angles = [90, 180, 270]
        # Defining how many times the image will be scaled
        scaling = 1
        # Iterating through defined rotation angles
        for angle in angles:
            # Rotation function
            R = cv2.getRotationMatrix2D(center, angle, scaling)
            # Applieing the rotated image to the original image
            rotation = cv2.warpAffine(image, R, (w, h))
            # Lowercase letters
            letters= string.ascii_lowercase
            # Joining 5 random letters
            st = ''.join(random.sample(letters,5))
            # Merging the location with 5 random letters
            name = location + st + '.jpg'
            # Saving the image
            cv2.imwrite(name, rotation)
            
if __name__ == '__main__': 
    # Executing the rot() function
    rot("C:\\Users\\KILE\\Desktop\\images\\")

