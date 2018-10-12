import csv
import cv2


def ReadDataset():
    images = []
    steerings = []
    with open("../data_end2end/driving_log.csv") as fileHandle:
        csvReader = csv.reader(fileHandle)
        for line in csvReader:
            # Read center image
            source_path_center = line[0]
            image_center = cv2.imread(source_path_center)
            image_rgb_center = cv2.cvtColor(image_center, cv2.COLOR_BGR2RGB)
            images.append(image_rgb_center)
            steering = float(line[3])
            steerings.append(steering)

            # Read left image
            source_path_left = line[1]
            image_left = cv2.imread(source_path_left)
            image_rgb_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
            images.append(image_rgb_left)
            steering_left = steering + 0.2
            steerings.append(steering_left)

            # Read right image
            source_path_right = line[2]
            image_right = cv2.imread(source_path_right)
            image_rgb_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)
            images.append(image_rgb_right)
            steering_right = steering - 0.2
            steerings.append(steering_right)

    # Return the lists for read images and steering measurements as a tuple.
    return images, steerings


if __name__ == "__main__":
    ReadDataset()
