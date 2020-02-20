import cv2
import numpy as np
import glob
 
img_array = []
photos = glob.glob('./*.png')
print photos[0].split('zdj')[1]
photos = sorted(photos, key=lambda x: int(x.split('zdj')[1].split('.')[0]))
for filename in photos:
    print filename
    img = cv2.imread(filename)
    if img is not None:
	    height, width, layers = img.shape
	    size = (width,height)
	    img_array.append(img)
 
 
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
