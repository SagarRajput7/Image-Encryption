import numpy as np 
from PIL import Image
import cv2 as cv
from matplotlib import pyplot as plt 
import matplotlib.image as mpimg

# img = np.zeros ((200,200),np.uint8)
img = cv.imread('cipher_duffing_lena.png')
b,g,r = cv.split(img)
# red = img[:,:,0]
# plt.hist(red.ravel(),256)
# plt.title('Red Histogram')
# plt.show()   

# cv.imshow("img",img)

# cv.imshow("b",b)
# cv.imshow("g",g)
# cv.imshow("r",r)
# plt.hist(img.ravel(),256,[0,256])
# plt.show()
plt.hist(b.ravel(),256,[0,256],color='blue')
plt.show()
plt.hist(g.ravel(),256,[0,256],color='green')
plt.show()
plt.hist(r.ravel(),256,[0,256],color='red')
plt.show()
# imgg = mpimg.imread('lennaMagicNew.jpg')
# plt.hist(imgg.ravel(),[0,256],density=False)
# plt.show()

# cv.waitkey(0)
cv.destroyAllWindows()