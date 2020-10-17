
#####用于显示相应图像的代码

import cv2
from google.colab.patches import cv2_imshow

img = cv2.imread('/content/ISIC_0000000.jpg')
label = cv2.imread('/content/ISIC_0000000_segmentation.png')
print(img.shape)
print(label.shape)
#cv2.imshow(img)  # disabled
#cv2_imshow(img)

[height, width]=img.shape[0:2]
size = (int(width*0.25), int(height*0.25))
shrink = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
shrink_label = cv2.resize(label, size, interpolation=cv2.INTER_AREA)
cv2_imshow(shrink)
cv2_imshow(shrink_label)

import numpy as np

##使用Opencv显示多个图像
imgs = np.hstack([shrink,shrink,shrink,shrink])
labels = np.hstack([shrink_label,shrink_label,shrink_label,shrink_label])
cv2_imshow(imgs)
cv2_imshow(labels)

import matplotlib.pyplot as plt

##可以查找调整大小的api

# 使用matplotlib展示多张图片
def matplotlib_multi_pic1():
    for i in range(4):
        img = cv2.imread('/content/ISIC_0000000.jpg')
        label = cv2.imread('/content/ISIC_0000000_segmentation.png')
        #title="title"+str(i+1)
        #行，列，索引
        plt.subplot(2,4,i+1)
        plt.imshow(shrink)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2,4,i+5)
        plt.imshow(shrink_label)
        #plt.title(title,fontsize=8)
        plt.xticks([])
        plt.yticks([])
    plt.show()
matplotlib_multi_pic1()


