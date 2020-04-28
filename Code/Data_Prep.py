import cv2
import glob
from Oxford_dataset.ReadCameraModel import ReadCameraModel
from Oxford_dataset.UndistortImage import UndistortImage
import numpy as np
import random


# function to get keypoints
def getKeypoints(old_img, current_image):

    # grayscale image
    old_gray_image = cv2.cvtColor(old_img, cv2.COLOR_BGR2GRAY)
    current_gray_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)

    # find keypoints with ORB
    kp1, des1 = orb.detectAndCompute(old_gray_image, None)
    kp2, des2 = orb.detectAndCompute(current_gray_image, None)

    # Match descriptors.
    matches = bf.match(des1, des2)
    # matches = bf.knnMatch(des1, des2, k=2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # draw only keypoints
    # key_image = cv2.drawKeypoints(img, kp, np.array([]), color=(0, 255, 0), flags=0)

    return matches


# function to create fundamental matrix
def fundamentalMatrix(kp, kp_old):
    i = random.sample(range(len(kp)+1), 8)
    i_old = random.sample(range(len(kp_old)+1), 8)
    return fMatrix


# main function
if __name__ == '__main__':

    # read dataset
    filenames = glob.glob("Oxford_dataset/stereo/centre/*.png")
    filenames.sort()

    # get camera parameters
    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('Oxford_dataset/model')

    # initiate STAR detector
    orb = cv2.ORB_create()

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for img in range(len(filenames)):
        old_frame = cv2.imread(filenames[img], 0)
        current_frame = cv2.imread(filenames[img+1], 0)

        # convert bayer image to color image
        old_color_frame = cv2.cvtColor(old_frame, cv2.COLOR_BayerGR2BGR)
        current_color_frame = cv2.cvtColor(current_frame, cv2.COLOR_BayerGR2BGR)

        # un-distort image
        old_undistorted_image = UndistortImage(old_color_frame, LUT)
        current_undistorted_image = UndistortImage(current_color_frame, LUT)

        # get keypoints using ORB
        match = getKeypoints(old_undistorted_image, current_undistorted_image)

        print(match[0].trainIdx)

        # fundamental matrix
        # F = fundamentalMatrix(current_keypoints, old_keypoints)

        # cv2.imshow('frame', keypoint_image)

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
