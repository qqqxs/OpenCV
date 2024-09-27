# -*- coding = utf -8 -*-
# @Time : 2022/7/18 19:24
# @File : 图像特征与特征匹配.py

import cv2
import numpy as np


def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('chessboard.jpg')
cv_show('img', img)
print('img.shape:', img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
print('dat.shape', dst.shape)

img[dst > 0.01 * dst.max()] = [0, 0, 255]
cv_show('dst', img)
