# opencv-python 下是 cv2
import cv2
import numpy
from matplotlib import pyplot
# imgobj= cv2.imread('../44_images-type/images/img2.jpg')
# imgobj= cv2.imread('../44_images-type/images/img4.jpg')
# cv2.namedWindow("啦啦")
# cv2.imshow("啦啦",imgobj)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

if __name__=='__main__':
    imgob1=cv2.imread("../44_images-type/images/img2.jpg")
    imgob2=cv2.imread("../44_images-type/images/img3.jpg")
    print(imgob1)# 是一个三维数组
    hist1=cv2.calcHist([imgob1],[0],None,[256],[0.0,255.0])
    hist2=cv2.calcHist([imgob2],[0],None,[256],[0.0,255.0])
    pyplot.plot(range(256),hist1,'r')
    pyplot.plot(range(256),hist2,'b')
    pyplot.show()
    cv2.imshow('img1',imgob1)
    cv2.imshow('img2',imgob2)
    cv2.waitKey(0)