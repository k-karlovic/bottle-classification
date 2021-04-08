# Adding modules
import cv2
import os 
import random
import string

def noise_re(location): 
    """Noise reduction for individual images""" 
    # Iterating through individual images in a defined location
    for imagename in os.listdir(location):
    	# Creating a source from the location and image name
        src = location + imagename 
        # Image upload
        image = cv2.imread(src)
        # Noise reduction function
        noise_reduction = cv2.fastNlMeansDenoising(image,None,20,7,21)
        # Lowercase letters
        letters = string.ascii_lowercase
        # Joining 5 random letters
        st = ''.join(random.sample(letters,5))
        # Merging the location with 5 random letters
        name = location + st + '.jpg'
        # Saving the image
        cv2.imwrite(name, noise_reduction)
        
if __name__ == '__main__':
	# Executing the noise_re() function
    noise_re("C:\\Users\\KILE\\Desktop\\images\\")
    
