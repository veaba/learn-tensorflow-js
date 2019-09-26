import numpy as np 
import cv2 
img = cv2.imread("../44_images-type/images/img3.jpg")
cv2.imshow("image",img)
k=cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows() #s esc不保存退出
elif k==ord('s'):
    cv2.imwrite("img3-1.jpg",img)
    cv2.destroyAllWindows()    