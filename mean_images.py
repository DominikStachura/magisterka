import pickle
import numpy as np
import matplotlib.pyplot as plt

images = pickle.load(open('class_image_dict.pickle', 'rb'))
images_class_0 = images[0]
images_class_1 = images[1]
images_class_2 = images[2]

mean_image_0 = np.zeros(images_class_0[0].shape)
mean_image_1 = np.zeros(images_class_0[0].shape)
mean_image_2 = np.zeros(images_class_0[0].shape)

for img in images_class_0:
    mean_image_0 += img / len(images_class_0)

for img in images_class_1:
    mean_image_1 += img / len(images_class_1)

for img in images_class_2:
    mean_image_2 += img / len(images_class_2)

fig = plt.figure()
ax1 = fig.add_subplot(3, 1, 1)
ax1.set_title(f'Class 0 - number of photos assigned -> {len(images_class_0)}')
ax1.imshow(mean_image_0.astype('int'))

ax2 = fig.add_subplot(3, 1, 2)
ax2.set_title(f'Class 1 - number of photos assigned -> {len(images_class_1)}')
ax2.imshow(mean_image_1.astype('int'))

ax3 = fig.add_subplot(3, 1, 3)
ax3.set_title(f'Class 2 - number of photos assigned -> {len(images_class_2)}')
ax3.imshow(mean_image_2.astype('int'))

fig.savefig('output.png')
