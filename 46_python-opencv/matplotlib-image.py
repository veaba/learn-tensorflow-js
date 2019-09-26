import numpy as np 
import cv2
from matplotlib import pyplot as plt 

img = cv2.imread("../44_images-type/images/img3.jpg")
plt.imshow(img,cmap="gray",interpolation="bicubic")
plt.xticks([],plt.yticks([]))
plt.show()