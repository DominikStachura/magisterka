from pathlib import Path
import cv2
import numpy as np


class LoadPhotos:
    def __init__(self, path=r'./', begin_with='', end_with='.png'):
        self.path = path
        self.file_name_pattern = f'{begin_with}*{end_with}'
        self.photos = []

    def load(self, x_size, y_size):
        """
        Return arrays of photos [x_size x y_size] loaded from path given in init
        """
        self.photos = []
        print('Loading photos')
        # path = Path(self.path) / Path(self.file_name_pattern)
        for file in Path(self.path).rglob(f'{self.file_name_pattern}'):
            file = str(file)
            photo = cv2.imread(file)
            photo = cv2.resize(photo, (x_size, y_size), interpolation=cv2.INTER_AREA)
            photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
            self.photos.append(photo)
        print('No photos loaded') if len(self.photos) == 0 else print('Loading finished')
        return np.array(self.photos)


if __name__ == "__main__":
    # test
    load = LoadPhotos(r'datasets', end_with='.png')
    p = load.load(32, 32)
