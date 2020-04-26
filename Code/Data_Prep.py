import cv2
import glob
from Oxford_dataset.ReadCameraModel import ReadCameraModel
from Oxford_dataset.UndistortImage import UndistortImage


# function to get keypoints
def getKeypoints(img):
    # grayscale image
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # intiate STAR detector
    orb = cv2.ORB()

    # find keypoints with ORB
    kp = orb.detect(gray_image, None)

    # draw only keypoints
    return cv2.drawKeypoints(img, kp, color=(0, 255, 0), flags=0)


# main function
if __name__ == '__main__':

    # read dataset
    filenames = glob.glob("Oxford_dataset/stereo/centre/*.png")
    filenames.sort()

    # get camera parameters
    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('Oxford_dataset/model')

    for image in filenames:
        frame = cv2.imread(image, 0)

        # convert bayer image to color image
        color_frame = cv2.cvtColor(frame, cv2.COLOR_BayerGR2BGR)

        # un-distort image
        undistorted_image = UndistortImage(color_frame, LUT)

        # get keypoints using ORB
        keypoint_image = getKeypoints(undistorted_image)

        cv2.imshow('frame', undistorted_image)

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
