import cv2
import matplotlib.pyplot as plt

img = cv2.imread('dog.jpg')
'''
print(img)
cv2.imshow('dog.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


cv_show('dog.jpg', img)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv_show('hsv', hsv)

print(img.shape)
img1 = cv2.imread('dog.jpg', cv2.IMREAD_GRAYSCALE)
print(img1)
print(img1.shape)
print(cv_show('dog.jpg', img1))
print(type(img1))
print(img1.size)
print(img1.dtype)

vc = cv2.VideoCapture('test.mp4')

if vc.isOpened():
    ret, frame = vc.read()
else:
    ret = False

while ret:
    temp, frame = vc.read()
    if frame is None:
        break
    if temp is True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('result', gray)
        if cv2.waitKey(10) & 0xFF is 27:
            break

vc.release()
cv2.destroyAllWindows()


dog = img[0:200, 0:200]
cv_show('dog', dog)

b, g, r = cv2.split(img)
print(b)
print(b.shape)

img = cv2.merge((b, g, r))
print(img.shape)

cur_img = img.copy()
cur_img[:, :, 0] = 0
cur_img[:, :, 1] = 0
cv_show('R', cur_img)

top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REPLICATE)
plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('replicate')
plt.show()


img_cat = cv2.imread('cat.jpg')
img_dog = cv2.imread('dog.jpg')

img_cat2 = img_cat + 10

print(img_cat[:5, :, 0])
print(img_cat2[:5, :, 0])
print((img_cat + img_cat2)[:5, :, 0])  # %256
print(cv2.add(img_cat, img_cat2)[:5, :, 0])


print(img_cat.shape)
img_dog = cv2.resize(img_dog, (500, 414))
print(img_dog.shape)
res = cv2.addWeighted(img_cat, 0.4, img_dog, 0.6, 0)
plt.imshow(res)
plt.show()

res = cv2.resize(img, (0, 0), fx=4, fy=4)
plt.imshow(res)
plt.show()
