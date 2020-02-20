import glob
from pathlib import Path
import cv2
import numpy as np


class LoadPhotos:
    def __init__(self, path=r'./', begin_with='', end_with=''):
        self.path = path
        self.file_name_pattern = f'{begin_with}*{end_with}'
        self.photos = []

    def load(self, x_size, y_size):
        print('Loading photos')
        path = Path(self.path) / Path(self.file_name_pattern)
        filenames = glob.glob(str(path))
        for file in filenames:
            photo = cv2.imread(file)
            photo = cv2.resize(photo, (x_size, y_size), interpolation=cv2.INTER_AREA)
            photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
            self.photos.append(photo)
            print('No photos loaded') if len(self.photos) == 0 else print('Loading finished')
        return np.array(self.photos)


if __name__ == "__main__":
    # test
    load = LoadPhotos(r'', begin_with='22019', end_with='.jpg')
