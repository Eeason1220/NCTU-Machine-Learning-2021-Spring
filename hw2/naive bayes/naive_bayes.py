import gzip
import numpy as np
import matplotlib.pyplot as plt
import math


def data_processing():
	train_image = open('./train-images.idx3-ubyte','rb')
	train_label = open('./train-labels.idx1-ubyte','rb')
	test_image = open('./t10k-images.idx3-ubyte','rb')
	test_label = open('./t10k-labels.idx1-ubyte','rb')
	train_image.read(16)
	train_label.read(8)
	test_image.read(16)
	test_label.read(8)
	return train_image,train_label,test_image,test_label


def discrete_print(attr):
	print("Imagination of numbers in Bayesian classifier:")
	for num in range(10):
		print(num, ":")
		for i in range(col_size):
			for j in range(row_size):
				black = 0
				white = 0
				for k in range(16):
					black+=attr[num][i*col_size+j][k]
					white+=attr[num][i*col_size+j][k+16]
				if (black>white):
					print('0',  end =" ")
				else:
					print('1',  end =" ")
			print("")
		print("")
		print("")


def continuous_print(mean):
	print("Imagination of numbers in Bayesian classifier:")
	for num in range(10):
		print(num, ":")
		for i in range(col_size):
			for j in range(row_size):
				for k in range(16):
					mean_num = mean[num][i*col_size+j]
				if (mean_num>128):
					print('1',  end =" ")
				else:
					print('0',  end =" ")
			print("")
		print("")
		print("")


def discrete_mode(train_image, train_label, test_image, test_label):
	labels = np.zeros((10,), dtype=np.int)
	attr = np.zeros((10, img_size, 32), dtype=np.int)
	pixel_total = np.zeros((10, img_size), dtype=np.int)
	for i in range(train_nums):
		num = int.from_bytes(train_label.read(1), byteorder='big')
		labels[num]+=1
		for pos in range(img_size):
			pixel = int.from_bytes(train_image.read(1), byteorder='big')
			attr[num][pos][int(pixel/8)]+=1
			pixel_total[num][pos]+=1

	correct = 0
	for i in range(test_nums):
		posterior = np.zeros((10,), dtype=np.int)
		target_num = int.from_bytes(test_label.read(1), byteorder='big')
		for num in range(10):
			posterior[num] += np.log(labels[num]/train_nums)    #prior
		for pos in range(img_size):
			pixel = int.from_bytes(test_image.read(1), byteorder='big')
			for num in range(10):
				likelihood = attr[num][pos][int(pixel/8)]/pixel_total[num][pos]
				if (likelihood==0):
					likelihood = 0.0000001
				posterior[num] += np.log(likelihood)
		predict = np.argmax(posterior)
		total = 0
		for num in range(10):
			total += posterior[num]
		if (predict == target_num):
			correct+=1
		print("Test data: ", i)
		print("Posterior (in log scale):")
		for num in range(10):
			print(num,": ", posterior[num]/total)
		print("Prediction:", predict," ,Ans: ", target_num)
		print("")
	print("error rate:", 1-(correct/10000))
	discrete_print(attr)


def continuous_mode(train_image, train_label, test_image, test_label):
	labels = np.zeros((10,), dtype=np.int)
	mean = np.zeros((10, img_size), dtype=np.int)
	mean2 = np.zeros((10, img_size), dtype=np.int)
	var = np.zeros((10, img_size), dtype=np.int)
	for i in range(train_nums):
		num = int.from_bytes(train_label.read(1), byteorder='big')
		labels[num]+=1
		for pos in range(img_size):
			pixel = int.from_bytes(train_image.read(1), byteorder='big')
			mean[num][pos] += pixel
			mean2[num][pos]+=pixel**2
	for num in range(10):
		for pos in range(img_size):
			mean[num][pos]/=labels[num]
			mean2[num][pos]/=labels[num]
			var[num][pos] = mean2[num][pos]-mean[num][pos]**2
			if (var[num][pos]==0):
				var[num][pos] = 1000
            
	correct = 0
	for i in range(test_nums):
		posterior = np.zeros((10,), dtype=np.int)
		target_num = int.from_bytes(test_label.read(1), byteorder='big')
		for num in range(10):
			posterior[num] += np.log(labels[num]/train_nums)
		for pos in range(img_size):
			pixel = int.from_bytes(test_image.read(1), byteorder='big')
			for num in range(10):
				likelihood = -0.5*np.log(2*math.pi*var[num][pos])-0.5*((pixel-mean[num][pos])**2)/var[num][pos]
				posterior[num] += likelihood
		predict = np.argmax(posterior)
		total = 0
		for num in range(10):
			total += posterior[num]
		if (predict == target_num):
			correct+=1
		print("Test data: ", i)
		print("Posterior (in log scale):")
		for num in range(10):
			print(num,": ", posterior[num]/total)
		print("Prediction:", predict," ,Ans: ", target_num)
		print("")
	print("error rate:", 1-(correct/10000))
	continuous_print(mean)


if __name__ == "__main__":

	train_nums = 60000
	test_nums = 10000
	row_size = 28
	col_size = 28
	img_size = row_size*col_size
	train_image, train_label, test_image, test_label = data_processing()
	
	print('discrete_mode:')
	discrete_mode(train_image, train_label, test_image, test_label)
	print('')
	print('continuous_mode')
	continuous_mode(train_image, train_label, test_image, test_label)
