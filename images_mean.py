import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path

pickles = glob.glob('pickle_outputs/new/2/*.pickle')
for pickle_dict in pickles:

    images = pickle.load(open(pickle_dict, 'rb'))
    mean_images = []
    for img_class, img_list in images.items():
        mean_images.append(sum(img / len(img_list) for img in img_list))

    fig = plt.figure()
    counter = 0
    for img_class, mean_image in enumerate(mean_images):
        if not isinstance(mean_image, int):
            counter += 1
            ax = fig.add_subplot(len(images), 1, counter)
            ax.set_title(f'Class {img_class} - number of photos assigned -> {len(images[img_class])}')
            ax.imshow(mean_image)

    fig.savefig(f'mean_images_output/new/2/{Path(pickle_dict).stem}.png')
