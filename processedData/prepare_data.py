import glob
import random
import time

import cv2
import numpy as np
import skimage as sk
import sklearn.feature_extraction
from skimage import feature


def random_rotation(image_array):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)


def random_noise(image_array):
    # add random noise to the image
    return sk.util.random_noise(image_array)


def horizontal_flip(image_array):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]


def compute_HOG(img_list, images, labels, class_label):
    # Loading positive class data
    for img_path in img_list:
        img = cv2.imread(img_path)
        if img is None:
            print(img_path)

        if class_label == 'pepsi':
            img1 = img
            img2 = random_rotation(img)
            img3 = random_noise(img)
            img4 = horizontal_flip(img)

            img1 = cv2.resize(img1, (100, 100))
            img2 = cv2.resize(img2, (100, 100))
            img3 = cv2.resize(img3, (100, 100))
            img4 = cv2.resize(img4, (100, 100))

            start = time.time()
            H1 = feature.hog(img1, orientations=9, pixels_per_cell=(10, 10),
                             cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=False)
            H2 = feature.hog(img2, orientations=9, pixels_per_cell=(10, 10),
                             cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=False)
            H3 = feature.hog(img3, orientations=9, pixels_per_cell=(10, 10),
                             cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=False)
            H4 = feature.hog(img4, orientations=9, pixels_per_cell=(10, 10),
                             cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=False)

            images.append(H1)
            images.append(H2)
            images.append(H3)
            images.append(H4)
            labels.append(class_label)
            labels.append(class_label)
            labels.append(class_label)
            labels.append(class_label)
        else:
            img = cv2.resize(img, (100, 100))
            img1 = sklearn.feature_extraction.image.extract_patches_2d(img, (70, 70), max_patches=3)
            img2 = sklearn.feature_extraction.image.extract_patches_2d(img, (40, 40), max_patches=3)

            for i in range(img1.shape[0]):
                img_tmp = img1[i, :, :]
                img_tmp = cv2.resize(img_tmp, (100, 100))
                H = feature.hog(img_tmp, orientations=9, pixels_per_cell=(10, 10),
                                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=False)
                images.append(H)
                labels.append(class_label)

            for i in range(img2.shape[0]):
                img_tmp = img2[i, :, :]
                img_tmp = cv2.resize(img_tmp, (100, 100))
                H = feature.hog(img_tmp, orientations=9, pixels_per_cell=(10, 10),
                                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=False)
                images.append(H)
                labels.append(class_label)

    return images, labels


# img_path = './images/pepsi-logotype-examples/pepsi+logo000001.png'
img_dir_path = '../images/pepsi-logotype-examples/*'

img_list = glob.glob(img_dir_path)

images = []
labels = []

images, labels = compute_HOG(img_list, images, labels, 'pepsi')

img_dir_path = '../images/None/*'

img_list = glob.glob(img_dir_path)

images, labels = compute_HOG(img_list, images, labels, 'None')

print('number of images: ', len(images))
dataset = {'images': images, 'labels': labels}
np.save('dataset5', dataset)
