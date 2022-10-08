import cv2
import numpy as np


def cv_show(name, images):  # 输出图像函数
    cv2.imshow(name, images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 读入模板图像
img = cv2.imread('ocr_a_reference.png')
cv_show('template', img)

ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show('ref', ref)

thresh = cv2.threshold(ref.copy(), 10, 255, cv2.THRESH_BINARY_INV)[1]
cv_show('thresh', thresh)  # 二值化

# 识别外边框 只保留终点
ref_, refs, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, refs, -1, (0, 0, 255), 3)
cv_show('img', img)

digits_sort = []  # 用来对识别到的边框进行排序
digits = {}  # 每一个数字对应一个模板

# 将识别到的轮廓放入数组中利用sort方法进行排序
for i in refs:
    (x, y, w, h) = cv2.boundingRect(i)
    digits_sort.append((x, y, w, h))

digits_sort.sort()

for (i, (x, y, w, h)) in enumerate(digits_sort):
    ROI = thresh[y: y + h, x: x + w]
    ROI = cv2.resize(ROI, (57, 88))
    # cv_show('roi', ROI)     # 显示每一个模板
    digits[i] = ROI

# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 读入信用卡图像
image = cv2.imread('credit_card_03.png')

cv_show('image', image)

image = cv2.resize(image, (516, 300))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show('gray', gray)

tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
cv_show('tophat', tophat)

gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = cv2.convertScaleAbs(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))  # 归一化
gradX = gradX.astype("uint8")
print(np.array(gradX).shape)
cv_show('gradX', gradX)

gradY = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
gradY = cv2.convertScaleAbs(gradY)
(minVal, maxVal) = (np.min(gradY), np.max(gradY))
gradY = (255 * ((gradY - minVal) / (maxVal - minVal)))  # 归一化
gradY = gradY.astype('uint8')
print(np.array(gradY).shape)
cv_show('gradY', gradY)
gradXY = cv2.addWeighted(gradX, 0.5, gradY, 0.5, 0)
cv_show('gradXY', gradXY)

gradXY = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)  # 闭操作
cv_show('gradXY', gradXY)

thresh = cv2.threshold(gradXY, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  # 二值化
cv_show('thresh', thresh)

thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel)  # 闭操作
cv_show('thresh', thresh)

thresh_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]  # 画边框
cv2.drawContours(image, thresh_, -1, (0, 0, 255), 3)
cv_show('image.copy', image)

locations = []  # 选取所需要的4个边框

for i, c in enumerate(thresh_):
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    if 2.5 < ar < 4.0:
        if (40 < w < 85) and (10 < h < 40):
            locations.append((x, y, w, h))
            ROI = thresh[y: y + h, x: x + w]  # 展示识别的边框
            cv_show('l', ROI)

locations = sorted(locations, key=lambda a: a[0])  # 对四个边框坐标进行一维排序

for (i, (gx, gy, gw, gh)) in enumerate(locations):
    ROI = thresh[gy: gy + gh, gx: gx + gw]  # 展示排序后识别的边框
    cv_show('ROI', ROI)

output = []
group_sort = []
results = []

for (i, (gX, gY, gW, gH)) in enumerate(locations):
    groupOutput = []  # 灰度图显示4个区域
    group = gray[gY - 5: gY + gH + 5, gX - 5:gX + gW + 5]
    cv_show('group', group)
    c = group.copy()

    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show('group', group)  # 二值化

    group_ = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]  # 1个区域中4个数字的外轮廓

    for c in group_:
        (ax, ay, aw, ah) = cv2.boundingRect(c)
        group_sort.append((ax, ay, aw, ah))

    group_sort.sort()
    print(group_sort)

    for (fx, fy, fw, fh) in group_sort:
        ROI = group[fy: fy + fh, fx: fx + fw]
        ROI = cv2.resize(ROI, (57, 88))
        cv_show('ROI', ROI)  # 显示每一个模板

        scores = []

        for digit, digitROI in digits.items():
            result = cv2.matchTemplate(ROI, digitROI, cv2.TM_CCOEFF)
            score = cv2.minMaxLoc(result)[1]
            scores.append(score)
        groupOutput.append(str(np.argmax(scores)))
        print(groupOutput)
    results.append(groupOutput)
    group_sort.clear()

    cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
    cv_show('image', image)
    print(results)
