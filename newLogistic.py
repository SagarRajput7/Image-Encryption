from PIL import Image
import numpy as np

def decimalToBinary(num):
    return format(num, '08b')

img = Image.open('Lenna.png')
image_array = np.array(img)

b, g, r = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]


def redChannel(r):
	rr = 3.99
	N = r.shape[0]*r.shape[1]
	x = .5 + np.zeros(N)
	for n in range(N-1):
	    x[n+1] = rr * x[n] * (1 - x[n])
	y = x.reshape(r.shape[0], r.shape[1])
	p = y*255
	# print(p.shape)
	z = p.astype('uint8')
	imgg2 = np.zeros(r.shape[0]*r.shape[1])
	imgg = imgg2.reshape(r.shape[0], r.shape[1])
	for i in range(r.shape[0]):
	    for j in range(r.shape[1]):
	        p = decimalToBinary(r[i][j])
	        key = decimalToBinary(z[i][j])
	        a = [int(x) for x in str(p)]
	        b = [int(x) for x in str(key)]
	        string = ""
	        for x in range(8):
	            string = string + str(a[x] ^ b[x])
	        imgg[i][j] = int(string[::-1], 2)

	return imgg

def blueChannel(b):
	rr = 3.99
	N = b.shape[0]*b.shape[1]
	x = .5 + np.zeros(N)
	for n in range(N-1):
	    x[n+1] = rr * x[n] * (1 - x[n])
	y = x.reshape(b.shape[0], b.shape[1])
	p = y*255
	# print(p.shape)
	z = p.astype('uint8')
	imgg2 = np.zeros(b.shape[0]*b.shape[1])
	imgg = imgg2.reshape(b.shape[0], b.shape[1])
	for i in range(b.shape[0]):
	    for j in range(b.shape[1]):
	        p = decimalToBinary(b[i][j])
	        key = decimalToBinary(z[i][j])
	        a = [int(x) for x in str(p)]
	        c = [int(x) for x in str(key)]
	        string = ""
	        for x in range(8):
	            string = string + str(a[x] ^ c[x])
	        imgg[i][j] = int(string[::-1], 2)

	return imgg

def greenChannel(g):
	rr = 3.99
	N = g.shape[0]*g.shape[1]
	x = .5 + np.zeros(N)
	for n in range(N-1):
	    x[n+1] = rr * x[n] * (1 - x[n])
	y = x.reshape(g.shape[0], g.shape[1])
	p = y*255
	# print(p.shape)
	z = p.astype('uint8')
	imgg2 = np.zeros(g.shape[0]*g.shape[1])
	imgg = imgg2.reshape(g.shape[0], g.shape[1])
	for i in range(g.shape[0]):
	    for j in range(g.shape[1]):
	        p = decimalToBinary(g[i][j])
	        key = decimalToBinary(z[i][j])
	        a = [int(x) for x in str(p)]
	        b = [int(x) for x in str(key)]
	        string = ""
	        for x in range(8):
	            string = string + str(a[x] ^ b[x])
	        imgg[i][j] = int(string[::-1], 2)

	return imgg


def MagicSqrShuffle(img_array):
	if max(img_array.shape[0],img_array.shape[1])%2==0:
		N=max(img_array.shape[0],img_array.shape[1])+1
	else:
		N=max(img_array.shape[0],img_array.shape[1])
	magic_square = np.zeros((N,N), dtype=float)
	DMkey=np.zeros(img_array.shape, dtype=float)
	EImg= np.zeros(img_array.shape, dtype=float)
	n=1
	i, j = 0, N//2
	while n <= N**2:
		magic_square[i, j] = n
		n += 1
		newi, newj = (i-1) % N, (j+1)% N
		if magic_square[newi, newj]:
			i += 1
		else:
			i, j = newi, newj
	for k in range(10):
		for i in range(0,N-1):
			for j in range(0,N-1):
				magic_square[i][j]=(magic_square[i][j]*3)%(N*N + 1)
	for i in range(0,N-1):
		for j in range(0,N-1):
			img_array[i][j]=(magic_square[i][j]+img_array[i][j])%256
	for i in range(0,N-1):
		for j in range(0,N-1):
			EImg[i][j]=((DMkey[i][j]*256)+img_array[i][j])-magic_square[i][j]

	return img_array

redArr = MagicSqrShuffle(redChannel(r))
blueArr = MagicSqrShuffle(blueChannel(b))
greenArr = MagicSqrShuffle(greenChannel(g))

# im = Image.fromarray(redChannel(r))
# im.show()
# im = Image.fromarray(blueChannel(b))
# im.show()
# im = Image.fromarray(greenChannel(g))
# im.show()

rgb = np.dstack((redArr,blueArr,greenArr))
im = Image.fromarray(np.uint8(rgb))
im.show()
file = im.save('lennaNewLogic.png')
