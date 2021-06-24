import numpy as np
import math
import numpy.linalg as linalg
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

data_size = 10000
rs = 1/(data_size)
rc = 1/(256*256)
cluster_num = 2
max_iteration = 40
img_name = 'image2.png'
img = cv2.imread(img_name)
img = img.reshape((-1, 3))
img = img.astype(float)

def kernel_function(img):      #O(n^2)
    pos = np.zeros((10000,2),dtype=int)
    for i in range(100):
        for j in range(100):
            pos[100*i+j][0] = i
            pos[100*i+j][1] = j
    pos_norm = np.sum(pos**2, axis = -1)
    M1 = pos_norm[:,None] + pos_norm[None,:] - 2 * np.dot(pos, pos.T)
    img_norm = np.sum(img**2, axis = -1)
    M2 = img_norm[:,None] + img_norm[None,:] - 2 * np.dot(img, img.T)
    kernel_mat = np.exp((-rs * M1) + (-rc * M2))
    return kernel_mat

def centroid_init(data, method):

    # get centroid by sampling from uniform distribution for k-dimention data
    if(method == 'random'):
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        diff = data_max - data_min
        centroid = np.random.rand(cluster_num, cluster_num)
        for i in range(cluster_num):
            centroid[i,:] = data_min[i] + diff[i] * centroid[i,:] 
        return centroid
    if (method == 'kmeans++'):
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        diff = data_max - data_min
        centroid = np.random.rand(cluster_num, cluster_num)
        for i in range(cluster_num):
            centroid[i,:] = data_min[i] + diff[i] * centroid[i,:] 
        for k in range(1, cluster_num):
            distance_vector = []
            for i in range(data_size):
                dist = []
                for cluster_idx in range(k):
                    val = np.sum((data[i,:] - centroid[cluster_idx,:])**2)
                    dist.append(val)
                min_dist = np.min(dist)
                distance_vector.append(min_dist)
            prob = distance_vector / np.sum(distance_vector)
            new_centroid = np.random.choice(range(data_size), p=prob)
            centroid[k,:] = data[new_centroid,:]
        return centroid



def clustering(data, centroid, labels):
    for i in range(data_size):
        dist_vec = []
        for k in range(cluster_num):
            dist = np.sum((data[i,:] -  centroid[k,:] )**2)
            dist_vec.append(dist)
        cluster_idx = np.argmin(dist_vec)
        labels[i] = cluster_idx
    return labels

def visualization(labels, iteration, method, cut):
	img = cv2.imread(img_name)
	color = [(101,67,54), (169,200,200), (3,155,230), (0,0,0), (255,255,0), (255,0,255), (0, 255, 255) ,(255,255,255)]
	'''color = []
	for k in range(cluster_num):
		color.append((0,0,0))
		for i in range(data_size):
			if(labels[i] == k):
				color[k] = (img[int(i/100), int(i%100)])
				break'''
	for i in range(100):
		for j in range(100):
			img[i,j] = color[labels[i*100+j]]
	cv2.imwrite('spectral2_'+method +'_cut='+cut+'_c=' + str(cluster_num)+'_it' + str(iteration)+'.png', img )

def get_mask(labels):
	mask = []
	for k in range(cluster_num):
		mask.append(np.where(labels!=k,0,1).reshape(-1,1))
	mask = np.array(mask)
	return mask

def count_cluster_num(labels):
	cluster_cnt = np.zeros((cluster_num), dtype=int)
	for i in range(data_size):
		cluster_cnt[labels[i]]+=1
	return cluster_cnt

def update_centroid(data, labels, centroid):
    mask = get_mask(labels)
    cluster_cnt = count_cluster_num(labels)
    for k in range(cluster_num):
        centroid[k,:] = (mask[k].T @ data) / cluster_cnt[k]
    return centroid

def laplacian_matrix(kernel_mat, cut):
    if (cut == 'ratio'):
        D = np.sum(kernel_mat, axis=1)
        L = D - kernel_mat
        return L
    if (cut == 'normalized'):
        D  = np.diag(np.sum(kernel_mat, axis=1))
        D_sqrt = np.diag(np.power(np.diag(D), -0.5))
        L_sym = np.identity(data_size) - D_sqrt @ kernel_mat @ D_sqrt
    return L_sym

def eigmat_calulate(mat):
    print('computing eigen...')
    eigenVal, eigenVec = linalg.eig(mat)
    idx = np.argsort(eigenVal)[1: cluster_num+1]
    U = eigenVec[:,idx].real.astype(np.float32)
    return U

def normalizing_rows(U):
    T = np.empty_like(U)
    for i in range(len(U[:,0])):
        denominator = math.sqrt(np.sum(U[i,:]**2))
        T[i,:] = U[i,:] / denominator
    return T

def error_cal(labels, pre_labels):
	diff = abs(labels - pre_labels)
	error = np.sum(diff)
	return error

def kmeans(data, cut, method):
    print(data)
    centroid = centroid_init(data, method)
    print('centroid:{}'.format(centroid))
    labels = np.zeros((data_size), dtype = int)
    print(labels)
    iteration = 0
    while(iteration < max_iteration):
        iteration +=1
        print('iteration:{}'.format(iteration))
        pre_labels = np.copy(labels)
        labels = clustering(data, centroid, labels)
        print(labels)
        error = error_cal(labels, pre_labels)
        print('Error:{}'.format(error))
        centroid = update_centroid(data, labels, centroid)
        visualization(labels, iteration, method, cut)
        if (error==0):
            eigen_visualization(labels, data, cut, method)
            break

def eigen_visualization(labels, data, cut, method):
    
    color = ['black',  'darkorange']
    for k in range(cluster_num):
        for i in range(data_size):
            if (labels[i]==k):
                plt.scatter(data[i,0], data[i,1], s=5, c=color[k])
    plt.savefig(cut + '_c='+str(cluster_num)+'_222_' + method +'_clustersize=' + str(k)+'.png')

if __name__== '__main__':

    methods = ['random', 'kmeans++']
    for method in methods:
        print('target clusters:{}'.format(cluster_num))
        kernel_mat = kernel_function(img)
        
        #ratio cut
        print('ratio cut')
        L = laplacian_matrix(kernel_mat, 'ratio')
        U = eigmat_calulate(L)
        kmeans(U, 'ratio cut', method)

        #normalized  cut
        print('normalized cut')
        L_sym = laplacian_matrix(kernel_mat, 'normalized')
        U = eigmat_calulate(L_sym)
        T = normalizing_rows(U)
        kmeans(U, 'normalized cut', method)