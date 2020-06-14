import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import PIL.Image as image
import PIL



def show_img(img):
    img = img * 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))


def multiply_pixel(img, size):
    if isinstance(img, PIL.Image.Image):
        pixels = [np.array(pixel) for pixel in list(img.getdata())]

    elif isinstance(img, np.ndarray):
        pixels = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                pixels.append(img[i, j, :])

    init_width = int(np.sqrt(len(pixels)))
    reshaped_pixels = np.array([np.array(size ** 2 * [pixel]).reshape(size, size, 3) for pixel in pixels])
    rows = []
    for j in range(init_width):
        rows.append(np.concatenate([reshaped_pixels[i] for i in range(3 * j, 3 * (j + 1))], axis=1))
    new_img = np.concatenate(np.array(rows), axis=0)
    return new_img

if __name__ == "__main__":

    pil = False

    all_filters = pickle.load(open('filters_output/new/2/filters_dict_0_001_200_3.pickle', 'rb'))

    # last one
    all_filters = list(all_filters.values())[-1]
    fig = plt.figure(figsize=(20., 20.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(11, 12),  # creates 2x2 grid of axes
                     axes_pad=0.3,  # pad between axes in inch.
                     )

    if pil:
        all_filters = [image.fromarray(i, 'RGB') for i in all_filters]

    for num, (ax, im) in enumerate(zip(grid, all_filters)):
        # Iterating over the grid returns the Axes.
        im = multiply_pixel(2*im, 4)
        ax.set_title(num)
        ax.set_axis_off()
        ax.imshow(im)


    plt.show()

