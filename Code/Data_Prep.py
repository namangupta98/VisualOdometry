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

    return matches, kp1, kp2


# function to create fundamental matrix
def fundamentalMatrix(points, f1, f2):
    x, y = np.zeros(8), np.zeros(8)
    x_, y_ = np.zeros(8), np.zeros(8)
    A, X, X_ = [], [], []

    # generating A matrix
    for i in range(8):
        x[i], y[i] = f1[points[i].queryIdx].pt[0], f1[points[i].queryIdx].pt[1]
        x_[i], y_[i] = f2[points[i].trainIdx].pt[0], f2[points[i].trainIdx].pt[1]

        A_rows = np.array([[x[i]*x_[i], x[i]*y_[i], x[i], y[i]*x_[i], y[i]*y_[i], y[i], x_[i], y_[i], 1]])
        X_col = np.array([[x[i]], [y[i]], [1]])
        X_col_ = np.array([[x_[i]], [y_[i]], [1]])

        X.append(X_col)
        X_.append(X_col_)
        A.append(A_rows)

    X = np.reshape(np.array(X).T, (3, 8))
    X_ = np.reshape(np.array(X_).T, (3, 8))
    A = np.reshape(A, (8, 9))

    [U, S, V] = np.linalg.svd(A)
    vx = V[:, 8]

    F = np.reshape(vx, (3, 3))
    F = np.round(F, 4)
    F[2, 2] = 0

    return F, X, X_


# function to get best fundamental matrix
def fRANSAC(points, kp1, kp2):

    ass_prob = 0

    for n in range(100):
        DMatch = random.sample(points, 8)
        inlier_count = 0

        # get fundamental matrix for sample
        F, x, x_ = fundamentalMatrix(DMatch, kp1, kp2)

        # epipolar constraint
        ep_const = x_.T @ F @ x

        for i in range(len(DMatch)):
            if ep_const[i, i] < 0:
                inlier_count += 1

        new_prob = inlier_count/len(DMatch)

        if new_prob >= ass_prob:
            ass_prob = new_prob
            bestF = F

    return bestF


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
        match, key1, key2 = getKeypoints(old_undistorted_image, current_undistorted_image)

        # fundamental matrix with RANSAC
        F = fRANSAC(match, key1, key2)

        # print(F)

        # cv2.imshow('frame', old_undistorted_image)

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
