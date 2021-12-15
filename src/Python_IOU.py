import os
import sys
import numpy as np
import cv2


img1_name = sys.argv[1]
img2_name = sys.argv[2]


img1 = cv2.imread(img1_name, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_name, cv2.IMREAD_GRAYSCALE)

white_co_inter = 0
white_co_union = 0


for i in range(len(img1)):
    for j in range((len(img1[0]))):
        if ((img1[i][j] == 255) and (img2[i][j] == 255)):
            white_co_inter += 1
        if ((img1[i][j] == 255) or (img2[i][j] == 255)):
            white_co_union += 1

print("We have a IOU of : ", (white_co_inter / white_co_union) * 100, "%. between ", img1_name, " and ", img2_name, ".")
