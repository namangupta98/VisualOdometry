import cv2
import glob


if __name__ == '__main__':

    # read dataset
    filenames = glob.glob("../Oxford_dataset/stereo/centre/*.png")
    filenames.sort()
    photos = [cv2.imread(img) for img in filenames]

    ctr = 0

    for image in photos:

        # show images
        cv2.imshow('image', image)
        ctr += 1
        print(ctr)

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
