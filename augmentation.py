from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import glob
import numpy as np
import os


class Augmentation:
    def __init__(self):
        self.augmented_data = []
        self.data_for_augmentation = []

    def read_data_for_augmentation(self, data_path=None):
        if data_path == None:
            print('Specify path to images you want to augment')
            return
        for img in glob.glob(f'{data_path}/*png'):
            self.data_for_augmentation.append(plt.imread(img))
        self.data_for_augmentation = np.array(self.data_for_augmentation)
        if len(self.data_for_augmentation) == 0:
            print('No images were found in the specified directory')

    def augment_data(self, rotation_range=10, zoom_range=0.1, horizontal_flip=False, path_to_save='.',
                     num_of_augmented_photos=10):
        if len(self.data_for_augmentation) == 0:
            print('Load images for augmentation first')
            return

        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        data_generator = ImageDataGenerator(rotation_range=rotation_range,
                                            zoom_range=zoom_range,
                                            horizontal_flip=horizontal_flip)

        data_generator.fit(self.data_for_augmentation)
        image_iterator = data_generator.flow(self.data_for_augmentation)
        for i in range(num_of_augmented_photos):
            plt.imsave(f'{path_to_save}/augmented{i}', image_iterator.next()[0])


augmentation = Augmentation()

augmentation.read_data_for_augmentation('data_for_augmentation')
augmentation.augment_data(path_to_save='augmeented_data')