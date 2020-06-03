import os
import cv2

for root, dirs, files in os.walk("datasets/new/"):
    for file in files:
        if '.jpg' in file:
                path = root+'/' + file
                img = cv2.imread(path)
                img = cv2.resize(img, (64, 64))
                cv2.imwrite(path, img)
