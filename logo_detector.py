import argparse
import pickle
import cv2
from skimage import feature

from logoDetection.utils import *

img_path = './images/testing-images/pepsi+products000137.jpg'
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default=img_path, help="Path to the image")
ap.add_argument("-s", "--scale", type=float, default=1.5, help="scale factor size")
args = vars(ap.parse_args())
# 11, 16, 23, 30, 29, 
image = cv2.imread(args['image'])
if image is None:
    exit('Image path is not valid. Please check!')


def detect_logo(image):
    (winW, winH) = (50, 50)

    model = pickle.load(open('learnedModels/learned5.sav', 'rb'))

    boxes = np.zeros((0, 4))
    # loop over the image pyramid
    for resized, scale in pyramid(image, scale=1.5):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=12, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            # Main process
            img = cv2.resize(window, (100, 100))
            H = feature.hog(img, orientations=9, pixels_per_cell=(10, 10),
                            cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=False)
            H = H.reshape(-1, 2916)

            clone = resized.copy()
            # circles = detect_circle(img)
            # if circles is not None:

            pred = model.predict(H)
            if pred == 'pepsi':
                boxes = np.vstack((boxes, np.array([x, y, x + winW, y + winH]).reshape(1, 4) * scale))

    boxes_nms = non_max_suppression_fast(boxes, 0.3)
    return boxes_nms


def show_imgs(image, boxes_nms):
    # final show
    for i in range(boxes_nms.shape[0]):
        rect = boxes_nms[i, :]
        cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 3)

    cv2.imshow("Final", image)
    cv2.waitKey(0)


def main():
    boxes_nms = detect_logo(image)
    with open('detectedBoxes.csv', 'a') as f:
        f.write("\nFor image at: {0}\n".format(img_path))
        for n, box in enumerate(boxes_nms):
            for i in box:
                f.write(str(i)); f.write(" ")
            f.write('\n')
        f.write("Number of Boxes: {0:d}\n".format(n+1))
    show_imgs(image, boxes_nms)


if __name__ == '__main__':
    main()
