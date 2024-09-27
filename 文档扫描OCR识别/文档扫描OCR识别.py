# -*- coding = utf -8 -*-
# @Time : 2022/7/18 19:51
# @File : 文档扫描OCR识别.py

import cv2
import numpy as np


def order_points(pts):
    # 一共4个坐标点
    rect = np.zeros((4, 2), dtype="float32")

    # 下面这个操作，是因为这四个点目前是乱序的，下面通过了一种巧妙的方式来找到对应位置
    # 左上和右下， 对于左上的这个点，(x,y)坐标和会最小， 对于右下这个点，(x,y)坐标和会最大，所以坐标求和，然后找最小和最大位置就是了
    # 按照顺序找到对应坐标0123分别是左上， 右上， 右下，左下
    s = pts.sum(axis=1)
    # 拿到左上和右下
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 右上和左下， 对于右上这个点，(x,y)坐标差会最小，因为x很大，y很小， 而左下这个点， x很小，y很大，所以坐标差会很大
    # 拿到右上和左下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    # 拿到正确的左上，右上， 右下，左下四个坐标点的位置
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算输入的w和h值 这里就是宽度和高度，计算方式就是欧几里得距离，坐标对应位置相减平方和开根号
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 变换后对应坐标位置
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 有了四组坐标，直接两个函数
    W = cv2.getPerspectiveTransform(rect, dst)  # 求透视变换矩阵
    warped = cv2.warpPerspective(image, W, (maxWidth, maxHeight))  # 透视变换

    # 返回变换后结果
    return warped


def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('receipt.jpg')
ratio = img.shape[0] / 500.0
orig = img.copy()
print(ratio)
cv_show('orig', orig)

img = cv2.resize(img, (700, 500))
cv_show('img', img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)
cv_show('edged', edged)

cnt = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
cnt = sorted(cnt, key=cv2.contourArea, reverse=True)[:5]

for c in cnt:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        cnt = approx
        break

cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
cv_show('Outline', img)


warped_1 = four_point_transform(orig, cnt.reshape(4, 2) * ratio)
warped_1 = cv2.cvtColor(warped_1, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(warped_1, 150, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite('scan.jpg', ref)
cv_show('original', cv2.resize(orig, (700, 500)))
cv_show('scanned', cv2.resize(ref, (700, 500)))

rows, cols = ref.shape[:2]
center = (cols / 2, rows / 2)  # 以图像中心为旋转中心
angle = 90  # 顺时针旋转90°
scale = 1  # 等比例旋转，即旋转后尺度不变

M = cv2.getRotationMatrix2D(center, angle, scale)  # 这里得到了一个旋转矩阵
rotated_img = cv2.warpAffine(ref, M, (cols, rows))

cv_show('rotated', cv2.resize(rotated_img, (900, 900)))
