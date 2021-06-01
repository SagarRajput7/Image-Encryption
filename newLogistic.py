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














# def lfsr2(seed, taps):
#     sr = seed
#     nbits = 8
#     while 1:
#         xor = 1
#         for t in range(1,taps):
#             if (sr & (1<<(t-1))) != 0:
#                 xor ^= 1
#         sr = (xor << nbits-1) + (sr >> 1)
#         yield xor, sr
#         if sr == seed:
#         	break
        

# # def KeyGen(N):
# # 	N=N*8
# # 	x=np.zeros((N,1),dtype=np.uint8)
# # 	r = 3.9999998
# # 	Xn=0
# # 	Xn_1 = 0.300001
# # 	for i in range(2,N):
# # 		Xn = 1-2*Xn_1*Xn_1
# # 		if (Xn > 0.0):
# # 			x[i-1]=1
# # 		else:
# # 			Xn_1 = Xn

# # 	key = np.zeros((int(N/8),1),dtype=np.uint8)
# # 	for i in range(1,int(N/8)):
# # 		for j in range(1,8):
# # 			key[i]=key[1]+x[i*j]*pow(2,(j-1))

# # 	return key 


# def toBinary(n):
# 	bnr = bin(int(n)).replace('0b','')
# 	x = bnr[::-1] 
# 	while len(x) < 8:
# 	    x += '0'
# 	bnr = x[::-1]
# 	return bnr

# def KeyGen(img):
# 	image_array = np.array(img)
# 	N = image_array.shape[1]
# 	x=0.300001+np.zeros((N,N),dtype=np.uint8)
# 	r = 3.9999998
# 	for i in range(N-1):
# 		x[i+1] = r * x[i] * (1 - x[i])
# 	y = x.reshape(image_array.shape[0], image_array.shape[1])
# 	key=y*255
# 	# print(key.shape)
# 	keyB = np.zeros((N,N),dtype=np.uint8)
# 	key2 = np.zeros((N,N),dtype=np.uint8)
# 	for i in range(N):
# 		for j in range(N):
# 			keyB[i][j] = toBinary(key[i][j])
# 	nbits = 8
# 	i=0
# 	for xor, sr in lfsr2(0b10111101,(8)):
# 		key2[i]=bin(2**nbits+sr)[3:]
# 		i+=1
# 	KEY =  np.bitwise_xor(key2,keyB)
# 	# print(key2[:N],keyB)
# 	return KEY

# def Encryp(image_array,KEY):
# 	P = np.bitwise_xor(image_array,KEY)
# 	return P




    
    


# K=KeyGen(img)
# print(K.shape)
# image_array=MagicSqrShuffle(img)
# # img2 = Image.fromarray(image_array, '1')
# # imgMagic = np.array(img2.convert('L'))
# # imag_array=MagicSqrShuffle(imgMagic)
# image = Image.fromarray(Encryp(image_array,K))
# image.show()
# file = image.save('lennaMagicNew.jpg')

		
	









	
	







