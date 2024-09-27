# -*- coding = utf -8 -*-
# @Time : 2022/7/10 14:25
# @File : 图像处理.py

import cv2  # opencv读取的格式是BGR
import numpy as np
import matplotlib.pyplot as plt  # Matplotlib是RGB


def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('cat.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img_gray.shape)

cv_show('cat', img)
cv_show('cat in gray', img_gray)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv_show('hsv', hsv)

ret1, thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
ret2, thresh2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
ret3, thresh3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
ret4, thresh4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
ret5, thresh5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)

titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])  # 不显示x，y坐标轴
plt.show()

img = cv2.imread('lenaNoise.png')
print(img.shape)
cv_show('img', img)

blur = cv2.blur(img, (3, 3))
cv_show('blur', blur)

box = cv2.boxFilter(img, -1, (3, 3), normalize=True)
cv_show('box', box)

gaussian = cv2.GaussianBlur(img, (5, 5), 1)
cv_show('gaussian', gaussian)

median = cv2.medianBlur(img, 5)
cv_show('median', median)

res = np.hstack((blur, gaussian, median))
cv_show('res', res)

img = cv2.imread('dige.png')
cv_show('img', img)

kernel = np.ones((3, 3), np.uint8)     # kernel是一个新的数组
erosion = cv2.erode(img, kernel, iterations=1)
cv_show('erosion', erosion)

dilate = cv2.dilate(erosion, kernel, iterations=1)
cv_show('dilate', dilate)

img = cv2.imread('pie.png')
cv_show('pie', img)
kernel = np.ones((30, 30), np.uint8)
erosion_1 = cv2.erode(img, kernel, iterations=1)
erosion_2 = cv2.erode(img, kernel, iterations=2)
erosion_3 = cv2.erode(img, kernel, iterations=3)
res = np.hstack((erosion_1, erosion_2, erosion_3))
cv_show('res', res)

dilate_1 = cv2.dilate(img, kernel, iterations=1)
dilate_2 = cv2.dilate(img, kernel, iterations=2)
dilate_3 = cv2.dilate(img, kernel, iterations=3)
res = np.hstack((dilate_1, dilate_2, dilate_3))
cv_show('res', res)

img = cv2.imread('dige.png')
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv_show('opening', opening)

closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv_show('closing', closing)

pie = cv2.imread('pie.png')
kernel = np.ones((7, 7), np.uint8)
dilate = cv2.dilate(pie, kernel, iterations=5)
erosion = cv2.erode(pie, kernel, iterations=5)

res = np.hstack((dilate, erosion))
cv_show('res', res)

gradient = cv2.morphologyEx(pie, cv2.MORPH_GRADIENT, kernel)
cv_show('gradient', gradient)

img = cv2.imread('dige.png')
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
cv_show('tophat', tophat)
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv_show('blackhat', blackhat)


img = cv2.imread('pie.png', cv2.IMREAD_GRAYSCALE)
cv_show('pie', img)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
cv_show('sobelx', sobelx)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
cv_show('sobelx', sobelx)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)
cv_show('sobely', sobely)

sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
cv_show('sobelxy', sobelxy)

sobelxy = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
cv_show('sobelxy', sobelxy)

img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
cv_show('lena', img)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
cv_show('sobelxy', sobelxy)

sobelxy = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
sobelxy = cv2.convertScaleAbs(sobelxy)
cv_show('sobelxy', sobelxy)


img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
cv_show('lena', img)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
scharry = cv2.convertScaleAbs(scharry)
scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

res = np.hstack((sobelxy, scharrxy, laplacian))
cv_show('res', res)


img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

v1 = cv2.Canny(img, 80, 150)
v2 = cv2.Canny(img, 50, 100)

res = np.hstack((v1, v2))
cv_show('res', res)

img = cv2.imread('car.png')
cv_show('car', img)

img = cv2.imread('car.png', cv2.IMREAD_GRAYSCALE)
cv_show('car', img)

v1 = cv2.Canny(img, 120, 250)
v2 = cv2.Canny(img, 50, 100)

res = np.hstack((v1, v2))
cv_show('res', res)


img = cv2.imread('AM.png')
cv_show('am', img)
print(img.shape)

up = cv2.pyrUp(img)
cv_show('up', up)
print(up.shape)

down = cv2.pyrDown(img)
cv_show('down', down)
print(down.shape)

up_down = cv2.pyrDown(up)
down_up = cv2.pyrUp(down)

res = np.hstack((img, up_down))
cv_show('res', res)

cv_show('img - up_down', img - up_down)

res = np.hstack((img, down_up))
cv_show('res', res)
cv_show('img - down_up', img - down_up)

res = np.hstack((img - down_up, img - up_down))
cv_show('lol', res)


img1 = cv2.imread('car.png')
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
ret1, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

img2 = cv2.imread('car.png', cv2.IMREAD_GRAYSCALE)
ret2, thresh2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
cv_show('l', thresh2)
res = np.hstack((thresh1, thresh2))
cv_show('res', res)

img = cv2.imread('car.png')
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv_show('thresh', thresh)

binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

draw_img = img.copy()
res = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2)
cv_show('res', res)

cnt = contours[0]
print(cv2.contourArea(cnt))
print(cv2.arcLength(cnt, True))


img = cv2.imread('contours2.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret3, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
binary3, contours3, hierarchy3 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours3[0]

draw_img = img.copy()
res = cv2.drawContours(draw_img, [cnt], -1, (0, 0, 255), 2)
cv_show('res', res)

epsilon = 0.1 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)

draw_img = img.copy()
res = cv2.drawContours(draw_img, [approx], -1, (0, 0, 255), 2)
cv_show('res', res)

img = cv2.imread('contours.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret11, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
binary1, contours1, hierarchy1 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours1[0]

x, y, w, h = cv2.boundingRect(cnt)
img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv_show('img', img)

area = cv2.contourArea(cnt)
x, y, w, h = cv2.boundingRect(cnt)
rect_area = w * h
extent = float(area) / rect_area
print('轮廓面积与边界矩形比', extent)

(x1, y1), radius = cv2.minEnclosingCircle(cnt)
center = (int(x1), int(y1))
radius = int(radius)
img = cv2.circle(img, center, radius, (0, 255, 0), 2)
cv_show('img', img)

img = cv2.imread('lena.jpg', 0)
img_float32 = np.float32(img)

dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()


img = cv2.imread('lena.jpg', 0)
img_float32 = np.float32(img)

dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)

mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow - 30:crow + 30, ccol - 30:crow + 30] = 1

fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()


img = cv2.imread('lena.jpg', 0)
img_float32 = np.float32(img)

dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)

mask = np.ones((rows, cols, 2), np.uint8)
mask[crow - 30:crow + 30, ccol - 30:crow + 30] = 0

fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Result'), plt.xticks([]), plt.yticks([])
plt.show()
