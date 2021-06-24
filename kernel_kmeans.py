import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
from scipy.spatial.distance import squareform, pdist

img_name = 'image1.png'
img = cv2.imread(img_name)
img_x = img.shape[0]
img_y = img.shape[1]
data_size = img_x*img_y
rs = 1/(data_size)
rc = 1/(256*256)
cluster_num = 4
max_iteration = 40
img = img.reshape((-1, 3))
img = img.astype(float)

def init(img, method):
	# initialize the centroids and labels(indicating what cluster is belonging to for each data point)
	# using 3 methods here: random, kmeans++ and mod method
	
	# RANDOM:
	# randomly assign a label sign to each data point
	
	if (method == 'random'):
		labels = np.random.randint(cluster_num, size = data_size)
		values, cluster_cnt = np.unique(labels, return_counts=True)
		visualization(labels, 0, 'random')
		return labels, cluster_cnt			 

	# KMEANS++:
	# 1. randomly pick up one data point as first centroid 
	# 2. for each data point x, compute the short distance between each discovered centroid and itself, set as D(x)
	# 3. the larger the D(x) is , x has more probability to be chosen as new centroid, 
		
	elif(method == 'kmeans++'):
		centroid = np.zeros((cluster_num, 2)) 
		centroid_1 = np.random.randint(0, 100, (1,2)) #randomly pick up a pixel in 100X100 image
		centroid[0,:] = centroid_1 
		for k in range(1, cluster_num): 
			distance_vector = []
			for i in range(img_x):
				for j in range(img_y):
					dist = []
					for cluster_idx in range(k):
						val = (i-centroid[cluster_idx, 0])**2 +  (j-centroid[cluster_idx, 1])**2 
						dist.append(val) 
					min_dist = np.min(dist) 
					distance_vector.append(min_dist)
			#prob is the probability of all pixel to be chosen as centroid
			prob = distance_vector / np.sum(distance_vector)
			#choose new_centroid based on prob
			new_centroid = np.random.choice(range(data_size), p=prob)	
			centroid[k,0] = int(new_centroid/100)
			centroid[k,1] = int(new_centroid%100)
		labels = clustering(centroid)
		values, cluster_cnt = np.unique(labels, return_counts=True)
		visualization(labels, 0, 'kmeans++')
		return labels, cluster_cnt

	elif (method == 'mod'):
		labels = np.zeros((data_size),dtype=int)
		for i in range(data_size):
			labels[i] = i%cluster_num
		values, cluster_cnt = np.unique(labels, return_counts=True)
		visualization(labels, 0, 'mod')
		return labels, cluster_cnt


def kernel_function(img):

	# compute kerlen function using two RBF kernel to consider
	# spatial similarity and color similarity at the same time.
    pos = np.zeros((data_size,2),dtype=np.uint8)
    for i in range(img_x):
        for j in range(img_y):
            pos[img_y*i+j][0] = i
            pos[img_y*i+j][1] = j
    pos_norm = np.sum(pos**2, axis = -1, dtype=np.uint8)
    M1 = (pos_norm[:,None] + pos_norm[None,:] - 2 * np.dot(pos, pos.T))
    img_norm = np.sum(img**2, axis = -1)
    M2 = img_norm[:,None] + img_norm[None,:] - 2 * np.dot(img, img.T)
    kernel_mat = (np.exp((-rs * M1) + (-rc * M2)))
    return kernel_mat

def count_cluster_num(labels):
	cluster_cnt = np.zeros((data_size),dtype=int)
	for i in range(data_size):
		cluster_cnt[labels[i]]+=1
	return cluster_cnt

def clustering(centroid):
	labels = np.zeros((data_size), dtype=int)
	for i in range(data_size):
		dist_vec = []
		pos = np.zeros((1,2), dtype=int)
		for k in range(cluster_num):
			pos[0,0] = int(i/img_y)
			pos[0,1] = int(i%img_y)
			dist = np.sum((pos[0,:] -  centroid[k,:] )**2)
			dist_vec.append(dist)
		cluster_idx = np.argmin(dist_vec)
		labels[i] = cluster_idx
	return labels

def get_mask(labels):

	#compute indicator vector foe each cluster
	mask = []
	for k in range(cluster_num):
		mask.append(np.where(labels!=k,0,1).reshape(-1,1))
	mask = np.array(mask)
	return mask

def term_2(cluster_cnt, kernel_mat, mask):
	output = np.zeros((data_size, cluster_num), dtype=float)
	for k in range(cluster_num):

		if(cluster_cnt[k]!=0):
			term2_vector =  (2/cluster_cnt[k])*kernel_mat.dot(mask[k])
		else:
			term2_vector =  (2)*kernel_mat.dot(mask[k])
		output[:,k] = term2_vector[:,0]
	return output

def term_3(cluster_cnt, kernel_mat, mask):
	output = np.zeros((cluster_num), dtype=float)
	for k in range(cluster_num):
		if(cluster_cnt[k]!= 0):
			term3_value = (1/cluster_cnt[k]**2) * mask[k].T @ kernel_mat @mask[k]
		else:
			term3_value = mask[k].T @ kernel_mat @mask[k]
		output[k] = term3_value
	return output
				
def error_cal(labels, pre_labels):
	diff = abs(labels - pre_labels)
	error = np.sum(diff)
	return error

def visualization(labels, iteration, method):
	img = cv2.imread(img_name)
	color = [(101,67,54), (169,200,200), (3,155,230), (0,0,0), (255,255,0), (255,0,255), (0, 255, 255) ,(255,255,255)]
	'''color = []
	for k in range(cluster_num):
		color.append((0,0,0))
		for i in range(data_size):
			if(labels[i] == k):
				color[k] = (img[int(i/100), int(i%100)])
				break'''
	for i in range(img_x):
		for j in range(img_y):
			img[i,j] = color[labels[i*img_x+j]]
	cv2.imwrite('discuss2'+method +'_c=' + str(cluster_num)+'_it' + str(iteration)+'.png', img )

def kernel_kmeans(img,method):

	
	# img: the original 10000X3 data

	# method: parameters indicating the way to initialize k-means clustering centroids
	# --random, kmeans++, mod method 
	labels, cluster_cnt= init(img, method)
	kernel_mat = kernel_function(img)    
	iteration = 0
	while(iteration < max_iteration):
		iteration +=1
		print('iteration:{}'.format(iteration))
		pre_labels = np.copy(labels)
		mask = get_mask(labels)
		term2 = term_2(cluster_cnt, kernel_mat, mask)
		term3 = term_3(cluster_cnt, kernel_mat, mask)
		for i in range(data_size):
			distance_vector = []
			for k in range(cluster_num):
				distance = kernel_mat[i,i] - term2[i,k] + term3[k]
				distance_vector.append(distance)
			predict_label = np.argmin(distance_vector)
			labels[i] = predict_label
		error = error_cal(labels, pre_labels)
		cluster_cnt = count_cluster_num(labels)
		print('Error:{}'.format(error))
		visualization(labels, iteration,method)
		if(error == 0):
			break
	return labels

if __name__== '__main__':
	methods = ['random', 'kmeans++', 'mod']
	print('target clusters:{}'.format(cluster_num))



	for method in methods:
		
		print('method:{}'.format(method))
		labels = kernel_kmeans(img, method)