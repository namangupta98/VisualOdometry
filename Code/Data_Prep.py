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

    return matches, kp1, kp2


# function to create fundamental matrix
def fundamentalMatrix(points, f1, f2):
    x, y = np.zeros(len(points)), np.zeros(len(points))
    x_, y_ = np.zeros(len(points)), np.zeros(len(points))
    A = []

    # generating A matrix
    for i in range(len(points)):
        x[i], y[i] = f1[points[i].queryIdx].pt[0], f1[points[i].queryIdx].pt[1]
        x_[i], y_[i] = f2[points[i].trainIdx].pt[0], f2[points[i].trainIdx].pt[1]

        A_rows = np.array([[x[i]*x_[i], x[i]*y_[i], x[i], y[i]*x_[i], y[i]*y_[i], y[i], x_[i], y_[i], 1]])
        A.append(A_rows)

    A = np.reshape(A, (8, 9))

    [U, S, V] = np.linalg.svd(A)
    vx = V[:, 8]

    F = np.reshape(vx, (3, 3))

    return F


# function to create matrix using keypoints
def keyMatrix(points, f1, f2):

    # key points storing in matrix
    x, y = np.zeros(len(points)), np.zeros(len(points))
    x_, y_ = np.zeros(len(points)), np.zeros(len(points))
    X, X_ = [], []

    # generating matrix
    for i in range(len(points)):
        x[i], y[i] = f1[points[i].queryIdx].pt[0], f1[points[i].queryIdx].pt[1]
        x_[i], y_[i] = f2[points[i].trainIdx].pt[0], f2[points[i].trainIdx].pt[1]

        X_col = np.array([[x[i]], [y[i]], [1]])
        X_col_ = np.array([[x_[i]], [y_[i]], [1]])

        X.append(X_col)
        X_.append(X_col_)

    X = np.reshape(np.array(X).T, (3, len(points)))
    X_ = np.reshape(np.array(X_).T, (3, len(points)))

    return X, X_


# function to get best fundamental matrix
def fRANSAC(points, kp1, kp2):

    x, x_ = keyMatrix(points, kp1, kp2)

    ass_prob = 0

    for n in range(100):
        DMatch = random.sample(points, 8)
        inlier_count = 0

        # get fundamental matrix for sample
        F = fundamentalMatrix(DMatch, kp1, kp2)

        # epi-polar constraint
        ep_const = x_.T @ F @ x

        for i in range(len(ep_const)):
            if ep_const[i, i] < 0:
                inlier_count += 1

        new_prob = inlier_count/len(points)

        if new_prob >= ass_prob:
            ass_prob = new_prob
            bestF = F

    return bestF


# function to get essential matrix
def essentialMatrix(KMat, F):
    E = KMat.T @ F @ KMat
    [U, S, V] = np.linalg.svd(E)
    E = U @ np.diag([1, 1, 0]) @ V.T
    return E


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

        # estimate fundamental matrix with RANSAC
        Fundamental_Matrix = fRANSAC(match, key1, key2)

        # essential matrix
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        Essential_Matrix = essentialMatrix(K, Fundamental_Matrix)

        # cv2.imshow('frame', old_undistorted_image)

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
