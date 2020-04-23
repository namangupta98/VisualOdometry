import cv2
import glob


if __name__ == '__main__':

    # read dataset
    filenames = glob.glob("../Oxford_dataset/stereo/centre/*.png")
    filenames.sort()

    for image in filenames:
        frame = cv2.imread(image)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
