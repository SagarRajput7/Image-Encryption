import numpy as np 
from PIL import Image
import cv2 as cv
from matplotlib import pyplot as plt 
import matplotlib.image as mpimg
img = cv.imread('cipher_duffing_lena.png')
b,g,r = cv.split(img)

plt.hist(b.ravel(),256,[0,256],color='blue')
plt.show()
plt.hist(g.ravel(),256,[0,256],color='green')
plt.show()
plt.hist(r.ravel(),256,[0,256],color='red')
plt.show()
cv.destroyAllWindows()
