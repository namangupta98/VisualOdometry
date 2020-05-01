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
    # matches = bf.match(des1, des2)
    # matches = bf.knnMatch(des1, des2, k=2)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # Sort them in the order of their distance.
    # matches = sorted(matches, key=lambda x: x.distance)

    return good, kp1, kp2


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

    F = np.reshape(V[:, -1], (3, 3))
    [U, S, V] = np.linalg.svd(F)

    F = U @ np.diag([S[0], S[1], 0]) @ V
    # F = np.round(F, 4)

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

    X = np.reshape(np.array(X).T, (3, len(points))).T
    X_ = np.reshape(np.array(X_).T, (3, len(points))).T

    return X, X_


# function to get best fundamental matrix
def fRANSAC(points, kp1, kp2):

    x, x_ = keyMatrix(points, kp1, kp2)

    ass_prob = 0

    for _ in range(100):
        DMatch = random.sample(points, 8)
        inlier_count = 0

        # get fundamental matrix for sample
        F = fundamentalMatrix(DMatch, kp1, kp2)

        # epi-polar constraint
        for i in range(len(x)):
            if abs(x_[i] @ F @ x[i].T) < 0.001:
                inlier_count += 1

        new_prob = inlier_count/len(points)

        if new_prob >= ass_prob:
            ass_prob = new_prob
            bestF = F

    return bestF, x, x_


# function to get essential matrix
def essentialMatrix(KMat, F):
    E = KMat.T @ F @ KMat
    [U, S, V] = np.linalg.svd(E)
    E = U @ np.diag([1, 1, 0]) @ V
    return E


# function to get camera poses
def cameraPoses(E):
    [U, _, V] = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # camera pose and rotation
    C1, C2, C3, C4 = U[:, 2], -U[:, 2], U[:, 2], -U[:, 2]
    R1, R2, R3, R4 = U @ W @ V.T, U @ W @ V.T, U @ W.T @ V.T, U @ W.T @ V.T
    C, R = np.array([[C1], [C2], [C3], [C4]]), [R1, R2, R3, R4]
    C = np.reshape(C, (4, 3)).T

    for i in range(len(R)):
        if np.linalg.det(R[i]) < 0:
            C[:, i] *= -1
            R[i] *= -1

    return C, R


# function for linear triangulation
def linearTriangulation(C, R, kp1, kp2):

    C = C.reshape((3, 1))

    H0 = np.eye(3, 4)
    H1 = np.hstack((R, C))

    rot = np.zeros((4, 3))
    trans = np.zeros((4, 1))

    Z = np.zeros((3, len(kp1)))

    pt1 = -np.eye(2, 3)
    pt2 = -np.eye(2, 3)

    # for i in range(len(kp1)):
    #     pt1[:, 2] = kp1[i, :]

    # return Z


# function to decompose essential matrix into translation and rotation matrix
def estimateCameraPose(E, kp1, kp2):

    # get camera poses
    C, R = cameraPoses(E)
    len((C[0]))
    Z = [0, 0, 0, 0]

    # triangulate 3D points using linear least square
    for i in range(len(C)):
        Z[i] = linearTriangulation(C[:, i], R[i], kp1, kp2)


    # return T, R


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
        old_frame = cv2.imread(filenames[img+18], 0)
        current_frame = cv2.imread(filenames[img+1+18], 0)

        # convert bayer image to color image
        old_color_frame = cv2.cvtColor(old_frame, cv2.COLOR_BayerGR2BGR)
        current_color_frame = cv2.cvtColor(current_frame, cv2.COLOR_BayerGR2BGR)

        # un-distort image
        old_undistorted_image = UndistortImage(old_color_frame, LUT)
        current_undistorted_image = UndistortImage(current_color_frame, LUT)

        # get key-points using ORB
        match, key1, key2 = getKeypoints(old_undistorted_image, current_undistorted_image)

        # estimate fundamental matrix with RANSAC
        Fundamental_Matrix, key1, key2 = fRANSAC(match, key1, key2)

        # essential matrix
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        Essential_Matrix = essentialMatrix(K, Fundamental_Matrix)

        # Get best translation and rotation matrix
        estimateCameraPose(Essential_Matrix, key1, key2)

        # cv2.imshow('frame', old_undistorted_image)

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
