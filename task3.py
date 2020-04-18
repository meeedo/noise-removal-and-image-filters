import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread(r"C:\Users\A.N\Desktop\test.png",1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

row,col = gray.shape
mean = 0 
var = 1000
sigma = var**0.5
gauss = np.array(gray.shape,np.float32)
gauss = np.random.normal(mean,sigma,(row,col))
gauss = gauss.reshape(row,col)
noisy_img = gray + gauss
#cv2.imshow('test',noisy_img)


avg_img = cv2.blur(noisy_img,(3,3))
med_img = cv2.medianBlur(noisy_img.astype(np.uint8),ksize=3)

img_sobx = cv2.Sobel(gray,cv2.CV_8U,1,0,ksize=3)
img_soby = cv2.Sobel(gray,cv2.CV_8U,0,1,ksize=3)
img_sobl = img_sobx+ + img_soby

kernalx = np.array([[-1,-1,-1],[0,0,0],[1,1,1,]])
kernaly = np.array([[-1,0,1],[-1,0,1],[-1,0,1,]])
img_perx = cv2.filter2D(gray,-1,kernalx)
img_pery = cv2.filter2D(gray,-1,kernaly)
img_prewitt = img_perx + img_pery


fig = plt.figure()

# show original image
fig.add_subplot(331)
plt.title(' origenal')
plt.set_cmap('gray')
plt.imshow(gray)

fig.add_subplot(332)
plt.title('noise')
plt.set_cmap('gray')
plt.imshow(noisy_img)


fig.add_subplot(333)
plt.title('subx')
plt.set_cmap('gray')
plt.imshow(img_sobx)

fig.add_subplot(334)
plt.title('soby')
plt.set_cmap('gray')
plt.imshow(img_soby)

fig.add_subplot(335)
plt.title('subl')
plt.set_cmap('gray')
plt.imshow(img_sobl)

fig.add_subplot(336)
plt.title('perx')
plt.set_cmap('gray')
plt.imshow(img_perx)

fig.add_subplot(337)
plt.title('pery')
plt.set_cmap('gray')
plt.imshow(img_pery)

fig.add_subplot(338)
plt.title('witt')
plt.set_cmap('gray')
plt.imshow(img_prewitt)

fig.add_subplot(339)
plt.title('average')
plt.set_cmap('gray')
plt.imshow(avg_img)


cv2.waitKey(0)
cv2.destroyAllWindows()
