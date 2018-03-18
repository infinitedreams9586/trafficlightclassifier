
# coding: utf-8

import cv2 # computer vision library
from IPython import get_ipython

import jupyterwork.helpers as helpers # helper functions
import jupyterwork.test_functions as test_functions

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().magic('matplotlib inline')

# Image data directories
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"

# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)


def show_img(index):
    image = IMAGE_LIST[index][0]
    label = IMAGE_LIST[index][1]
    print("Shape: (height, width, channel)", image.shape, " Label:", label)
    plt.imshow(image)

# This function should take in an RGB image and return a new, standardized version
def standardize_input(image):
    standard_im = np.copy(image)
    standard_im = cv2.resize(standard_im, (32, 32))
    return standard_im

def one_hot_encode(label):
    label_types = ['red', 'yellow', 'green']
    # Create a vector of 0's that is the length of the number of classes (3)
    one_hot_encoded = [0] * len(label_types)

    # Set the index of the class number to 1
    one_hot_encoded[label_types.index(label)] = 1 

    return one_hot_encoded

def standardize(image_list):
    
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))
        
    return standard_list

# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)

def show_std_img(index):
    image = STANDARDIZED_LIST[index][0]
    label = STANDARDIZED_LIST[index][1]
    print("Shape: (height, width, channel)", image.shape, " Label:", label)
    plt.imshow(image)
    
def get_example_by_label(label):
    result = [0, 0, 0]
    if label == 'red':
        result = [1, 0, 0]
    elif label == 'yellow':
        result = [0, 1, 0]
    elif label == 'green':
        result = [0, 0, 1]
    for img in STANDARDIZED_LIST:
        if img[1] == result:
            return img

def show_image(img):
    print("Shape: (height, width, channel)", img[0].shape, " Label:", img[1])
    plt.imshow(img[0])

def create_hsv_feature(rgb_image):
    import math
    feature = [0, 0, 0]
    ## TODO: Convert image to HSV color space
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    # HSV channels
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    
    # Add up all the pixel values in the H channel
    sum_h = np.sum(hsv[:,:,0])
    area = hsv.shape[0] * hsv.shape[1]
    avg_h = sum_h / area
    
    # Add up all the pixel values in the S channel
    sum_s = np.sum(hsv[:,:,1])
    avg_s = sum_s / area
    
    # Add up all the pixel values in the V channel
    sum_v = np.sum(hsv[:,:,2])
    avg_v = sum_v / area
    
    ## TODO: Create and return a feature value and/or vector
    feature = [math.floor(avg_h), math.floor(avg_s), math.floor(avg_v)]
    #feature = [sum_h, sum_s, sum_v]
    return feature

def create_red_mask(rgb_image):
    # RED mask
    lower_red = np.array([180,50,50]) 
    upper_red = np.array([255,180,180])
    mask = cv2.inRange(rgb_image, lower_red, upper_red)
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(rgb_image, rgb_image, mask= mask)
    return res

def create_yellow_mask(rgb_image):
    # YELLOW mask
    lower_yellow = np.array([200,200,150]) 
    upper_yellow = np.array([255,255,200])
    mask = cv2.inRange(rgb_image, lower_yellow, upper_yellow)
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(rgb_image, rgb_image, mask= mask)
    return res

def create_green_mask(rgb_image):
    # GREEN mask
    lower_green = np.array([120,128,130])
    upper_green = np.array([150,255,140])
    mask = cv2.inRange(rgb_image, lower_green, upper_green)
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(rgb_image, rgb_image, mask= mask)
    return res

def crop_and_blur_image(image):
    row = 5
    col = 10
    img = image.copy()
    img = img[row:-row, col:-col, :]
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

# (Optional) Add more image analysis and create more features
def create_rgb_feature(image):
    import math
    rgb_image = crop_and_blur_image(image.copy())
   
    # RGB channels
    r = rgb_image[:,:,0]
    g = rgb_image[:,:,1]
    b = rgb_image[:,:,2]
    
    # Add up all the pixel values in the R channel
    sum_r = np.sum(rgb_image[:,:,0])
    area = rgb_image.shape[0] * rgb_image.shape[1]
    avg_r = sum_r / area
    
    # Add up all the pixel values in the G channel
    sum_g = np.sum(rgb_image[:,:,1])
    avg_g = sum_g / area
    
    # Add up all the pixel values in the B channel
    sum_b = np.sum(rgb_image[:,:,2])
    avg_b = sum_b / area
    
    ## TODO: Create and return a feature value and/or vector
    feature = [math.floor(avg_r), math.floor(avg_g), math.floor(avg_g)]
    #feature = [sum_r, sum_g, sum_b]
    return feature

def get_brightest_spot(image):
    rgb_image = crop_and_blur_image(image.copy())
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    return maxLoc

# This function should take in RGB image input
# Analyze that image using your feature creation code and output a one-hot encoded label
def estimate_label_based_on_hsv(rgb_image):
    predicted_label = [0, 0, 0]
    red_masked_hsv = [0, 0, 0]
    yellow_masked_hsv = [0, 0, 0]
    green_masked_hsv = [0, 0, 0]
    
    rgb_image = crop_and_blur_image(rgb_image)
    # create masked images for original image
    red_masked = create_red_mask(rgb_image)  # red masked
    yellow_masked = create_yellow_mask(rgb_image) # yellow masked
    green_masked = create_green_mask(rgb_image) # green masked
    
    original_hsv = create_hsv_feature(rgb_image)
    
     # get HSV feature's h values for all masked images
    red_masked_hsv_h = create_hsv_feature(red_masked)[0]
    yellow_masked_hsv_h = create_hsv_feature(yellow_masked)[0]
    green_masked_hsv_h = create_hsv_feature(green_masked)[0]
    
     # get HSV feature's s values for all masked images
    red_masked_hsv_s = create_hsv_feature(red_masked)[1]
    yellow_masked_hsv_s = create_hsv_feature(yellow_masked)[1]
    green_masked_hsv_s = create_hsv_feature(green_masked)[1]
    
    # get HSV feature's v values for all masked images
    red_masked_hsv_v = create_hsv_feature(red_masked)[2]
    yellow_masked_hsv_v = create_hsv_feature(yellow_masked)[2]
    green_masked_hsv_v = create_hsv_feature(green_masked)[2]
    
    # compare v values
    if red_masked_hsv_v >= yellow_masked_hsv_v and red_masked_hsv_v >= green_masked_hsv_v:
        predicted_label[0] = 1  # predict red
    if yellow_masked_hsv_v > red_masked_hsv_v and yellow_masked_hsv_v >= green_masked_hsv_v:
        predicted_label[1] = 1  # predict yellow
    if green_masked_hsv_v > red_masked_hsv_v and green_masked_hsv_v > yellow_masked_hsv_v:
        predicted_label[2] = 1  # predict green
        if green_masked_hsv_s < red_masked_hsv_s:
            predicted_label[0] = 1 # predict red
            predicted_label[2] = 0 # remove green prediction

    return predicted_label, original_hsv

def estimate_label_based_on_rgb(rgb_image):
    predicted_label = [0, 0, 0]
    l = create_rgb_feature(rgb_image)
    r, g, b = l[0], l[1], l[2]
    if r > g and r > b:
        predicted_label[0] = 1 #red
    elif g > r and b > r and (g == b):
        predicted_label[2] = 1 #green
    elif  r > g and r > b and (g == b):
        predicted_label[1] = 1 #yellow
    return predicted_label, [r, g, b]

def estimate_label_based_on_brightspot_location(rgb_image):
    predicted_label = [0, 0, 0]
    bright_spot = get_brightest_spot(rgb_image)
    if bright_spot[1] <= 10:
        predicted_label[0] = 1 #red
    elif bright_spot[1] >= 11 and bright_spot[1] < 18:
        predicted_label[1] = 1 #yellow
    elif bright_spot[1] >= 18:
        predicted_label[2] = 1 #green
    return predicted_label, bright_spot
    

# CLASSIFIER --------
def estimate_label(rgb_image):
    
    ## TODO: Extract feature(s) from the RGB image and use those features to
    ## classify the image and output a one-hot encoded label
    result_hsv, hsv = estimate_label_based_on_hsv(rgb_image)
    result_rgb, rgb = estimate_label_based_on_rgb(rgb_image)
    result_loc, loc = estimate_label_based_on_brightspot_location(rgb_image)
    
    predicted_label = [x+y+z for x, y, z in zip(result_hsv, result_rgb, result_loc)]
    result = [0, 0, 0]
    max_val = max(predicted_label)
    if max_val != 0:
        for idx, label in enumerate(predicted_label):
#             print(idx, label, max_val, label/max_val)
            if (label / max_val) == 1.0:
                result[idx] = 1
                break
                
    return result   

# Using the load_dataset function in helpers.py
# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)

def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels 
        
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
#             print(predicted_label, true_label)
#             print('-----------')
            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))

# Importing the tests
tests = test_functions.Tests()

if(len(MISCLASSIFIED) > 0):
    # Test code for one_hot_encode function
    tests.test_red_as_green(MISCLASSIFIED)
else:
    print("MISCLASSIFIED may not have been populated with images.")


# In[305]:


number_of_misclassified = len(MISCLASSIFIED)
i = 0

misclassified_green = []

for item in range(number_of_misclassified ):
    selected_image = MISCLASSIFIED[item]
    predicted_label, true_label = selected_image[1], selected_image[2]    
    #print(predicted_label, true_label)

    if true_label == one_hot_encode("red") and predicted_label == one_hot_encode("green"):
        misclassified_green.append(selected_image)
        print("where in misclassified list:", item)
        i+=1

print("number of red lights mistaken as green:", i, " of ", number_of_misclassified, 
       "misclassified images")
