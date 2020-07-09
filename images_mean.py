import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from string import  ascii_lowercase


def generate_mean_images(images, output_file_name):
    mean_images = []
    for img_class, img_list in images.items():
        mean_images.append((img_class, sum(img / len(img_list) for img in img_list)))

    fig = plt.figure(figsize=(20, 20))
    counter = 0
    for (img_class, mean_image), annotation in zip(sorted(mean_images), ascii_lowercase):
        if not isinstance(mean_image, int):
            counter += 1
            ax = fig.add_subplot(1, len(images), counter)
            # ax.set_title(f'Class {img_class} - number of photos assigned -> {len(images[img_class])}')
            # ax.set_title(f'{img_class}', y=-0.1, fontsize=25)
            ax.set_title(f'{annotation})', y=-0.15, fontsize=25)
            ax.axis('off')
            ax.imshow(mean_image)
    # fig.tight_layout(pad=3)
    fig.subplots_adjust(hspace=0.6)
    fig.savefig(output_file_name)


if __name__ == '__main__':
    # pickles = glob.glob('pickle_outputs/new/mnist/class_image_dict_manually_generated.pickle')
    pickles = glob.glob('pickle_outputs/new/granulacja/*.pickle')

    for pickle_dict in pickles:
        images = pickle.load(open(pickle_dict, 'rb'))
        name = f'mean_images_output/new/granulacja/{Path(pickle_dict).stem}.png'
        generate_mean_images(images, name)

