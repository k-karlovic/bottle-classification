# Adding modules
import cv2
import os 
import random
import string
  
def blurred(location): 
    """Blurring individual images""" 
    # Iterating through individual images in a defined location      
    for imagename in os.listdir(location): 
        # Creating a source from the location and image name
        src = location + imagename
        # Image upload
        image = cv2.imread(src)
        # Blurring function
        blurred_image = cv2.medianBlur(image,5)
        # Lowercase letters
        letters = string.ascii_lowercase
        # Joining 5 random letters
        st = ''.join(random.sample(letters,5))
        # Merging the location with 5 random letters
        name = location + st + '.jpg'
        # Saving the image
        cv2.imwrite(name, blurred_image)
        

if __name__ == '__main__': 
	# Executing the blurred() function
    blurred("C:\\Users\\KILE\\Desktop\\images\\")
    
