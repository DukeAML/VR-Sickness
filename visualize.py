import cv2
import numpy as np
import glob

img_array = []
# filename_array = []
# for filename in glob.glob('./sharks_heatmaps/*.png'):
#     filename_array.append(filename)
#
# filename_array.sort()
for i in range(5, 1500):
    filename = "./sharks_heatmaps/" + str(i) + ".png"
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)
    print(filename)

out = cv2.VideoWriter('shark.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    if i%100 == 0:
        print(i)
    out.write(img_array[i])
out.release()