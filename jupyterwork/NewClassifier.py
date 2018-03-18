# coding: utf-8

import cv2 # computer vision library
import jupyterwork.helpers as helpers # helper functions
import math
import numpy as np
import matplotlib.pyplot as plt
plt.interactive(True)

# Image data directories
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"

def display_img(image):
    plt.imshow(image)

def show_img_from_list(color, IMAGE_LIST):
    image = IMAGE_LIST[0][0]
    label = IMAGE_LIST[0][1]
    for img in IMAGE_LIST:
        if img[1] == color:
            image = img[0]
            label = img[1]
            break
    print("Shape: (height, width, channel)", image.shape, " Label:", label)
    plt.imshow(image)

def crop_image(image):
    row = 4
    col = 6
    img = image.copy()
    img = img[row:-row, col:-col, :]
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

# This function should take in an RGB image and return a new, standardized version
def standardize_input(image):
    standard_im = np.copy(image)
    standard_im = cv2.resize(standard_im, (32, 32))
    cropped = crop_image(standard_im)
    return cropped

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

        # Append the image, and it's one hot encoded label to the full,
        # processed list of image data
        standard_list.append((standardized_im, one_hot_label))

    return standard_list

def slice_image(image):
    img = image.copy()
    upper = img[0:7, :, :]
    middle = img[8:15, :, :]
    lower = img[16:24, :, :]
    return upper, middle, lower

def get_avg_v(rgb_image):
    feature = [0, 0, 0]
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # calculate image area
    area = hsv.shape[0] * hsv.shape[1]

    # Add up all the pixel values in the V channel
    sum_v = np.sum(hsv[:, :, 2])
    avg_v = sum_v / area

    return math.floor(avg_v)

def classifier(image):
    upper, middle, lower = slice_image(image)
    upper_hv = get_avg_v(upper)
    middle_hv = get_avg_v(middle)
    lower_hv = get_avg_v(lower)

    max_v = max(upper_hv, middle_hv, lower_hv)
    result = [0, 0, 0]
    if max_v != 0:
        for idx, item in enumerate([upper_hv, middle_hv, lower_hv]):
            if item / max_v == 1:
                result[idx] = 1
                break
    else:
        result = [1, 0, 0]  # default red
    return result

def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:
        # Get true data
        im = image[0]
        true_label = image[1]
        assert (len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = classifier(im)
        assert (len(predicted_label) == 3), "The predicted_label is not the expected " \
                                    "length (3)."

        # Compare true and predicted labels

        if (predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))

    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


def main():
    # load images into image list
    IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)
    IMAGE_LIST_TEST = helpers.load_dataset(IMAGE_DIR_TEST)
    #
    # show_img_from_list('red', IMAGE_LIST)
    # show_img_from_list('yellow', IMAGE_LIST)
    # show_img_from_list('green', IMAGE_LIST)

    # Standardize all training images
    STANDARDIZED_LIST = standardize(IMAGE_LIST)
    STANDARDIZED_TEST_LIST = standardize(IMAGE_LIST_TEST)


    plt.imshow(STANDARDIZED_TEST_LIST[0][0])
    # Find all misclassified images in a given test set
    MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

    # Accuracy calculations
    total = len(STANDARDIZED_TEST_LIST)
    num_correct = total - len(MISCLASSIFIED)
    accuracy = num_correct / total

    print('Accuracy: ' + str(accuracy))
    print("Number of misclassified images = " + str(
        len(MISCLASSIFIED)) + ' out of ' + str(total))

    number_of_misclassified = len(MISCLASSIFIED)
    i = 0

    misclassified_green = []

    for item in range(number_of_misclassified):
        selected_image = MISCLASSIFIED[item]
        predicted_label, true_label = selected_image[1], selected_image[2]

        if true_label == one_hot_encode("red") and predicted_label == one_hot_encode("green"):
            misclassified_green.append(selected_image)
            print("---->")
            print("where in misclassified list:", item)
            i += 1
        #print(predicted_label, true_label, hv)
    print("number of red lights mistaken as green:", i, " of ",
          number_of_misclassified,
          "misclassified images")


if __name__ == '__main__':
    main()
