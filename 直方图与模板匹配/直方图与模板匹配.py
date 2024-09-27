# -*- coding = utf -8 -*-
# @Time : 2022/7/12 14:30
# @File : 直方图与模板匹配.py

import cv2  # opencv读取的格式是BGR
import numpy as np
import matplotlib.pyplot as plt  # Matplotlib是RGB


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

'''
img = cv2.imread('lena.jpg', 0)
template = cv2.imread('face.jpg', 0)
h, w = template.shape[:2]
print(img.shape)
print(template.shape)

res = cv2.matchTemplate(img, template, 0)
print(res.shape)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
print(min_val)
print(max_val)
print(min_loc)
print(max_loc)

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img2 = img.copy()
    method = eval(meth)
    print(method)

    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottem_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img2, top_left, bottem_right, (0, 0, 255), 2)

    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img2, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()

img_rgb = cv2.imread('mario.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('mario_coin.jpg', 0)
h1, w1 = template.shape[:2]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

thresh = 0.8
loc = np.where(res >= thresh)

for pt in zip(*loc[::-1]):
    bottem_right = (pt[0] + w1, pt[1] + h1)
    cv2.rectangle(img_rgb, pt, bottem_right, (0, 0, 255), 2)

cv_show('img_rgb', img_rgb)


img = cv2.imread('cat.jpg', 0)

hist = cv2.calcHist(img, [0], None, [256], [0, 256])
print(hist.shape)

plt.hist(img.ravel(), 256)
plt.show()

img = cv2.imread('cat.jpg')
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.show()


img = cv2.imread('cat.jpg')
mask = np.zeros(img.shape[:2], np.uint8)
print(mask.shape)
mask[100:300, 100:400] = 255
cv_show('mask', mask)

img = cv2.imread('cat.jpg', 0)
cv_show('img', img)

masked_img = cv2.bitwise_and(img, img, mask=mask)
cv_show('masked_img', masked_img)

hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])

plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0, 256])


img = cv2.imread('clahe.jpg', 0)
equ = cv2.equalizeHist(img)
res = np.hstack((img, equ))
cv_show('res', res)

clach = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
res_clahe = clach.apply(img)
res = np.hstack((img, equ, res_clahe))
cv_show('res', res)
'''