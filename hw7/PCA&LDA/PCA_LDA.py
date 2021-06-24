import numpy as np
import cv2 
import os
import re
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.distance import cdist

train_size = 135
test_size = 30
total_size = train_size + test_size
KNN_K = 5
shape = (50,50)



def dataLoader():
    path_name = './Yale_Face_Database/Training/'
    files = os.listdir(path_name)
    train_data = []
    train_label = []
    for file in files:
        label = re.search(r'\d+',file)
        train_label.append(label.group())
        img = cv2.imread(path_name +'/'+file,-1)
        img = cv2.resize(img, shape)
        train_data.append(img.reshape((-1)))
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    
    path_name = './Yale_Face_Database/Testing/'
    files = os.listdir(path_name)
    test_data = []
    test_label = []
    for file in files:
        label = re.search(r'\d+',file)
        test_label.append(label.group())
        img = cv2.imread(path_name +'/'+file,-1)
        img = cv2.resize(img, shape)
        test_data.append(img.reshape((-1)))
    test_data = np.array(test_data)
    test_label = np.array(test_label)
    
    return train_data, train_label, test_data, test_label



def RBF_kernel(X):   #compute kernel(gram) matrix, which use rational quadratic kernel as kernel function
    alpha = 0.00001
    kernel = np.zeros((len(X),len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            value = np.exp(-alpha * (np.sum((X[i,:]-X[j,:])**2)))
            kernel[i,j] = value
    return kernel



def poly_kernel(X):
    d = 2
    kernel = np.zeros((len(X),len(X)))
    for i in range(len(X[:,0])):
        for j in range(len(X)):
            value = (X[i].dot(X[j]))**d
            kernel[i,j] = value
    return kernel



def linear_kernel(X):
    kernel = X @ X.T
    return kernel



def tanh_kernel(X):
    kernel = np.zeros((len(X),len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            value = np.tanh(X[i].dot(X[j]))
            kernel[i,j] = value
    return kernel
    
    

def clustering_by_label(data_set, labels):
    label_list = np.unique(labels)
    cluster_num = len(label_list)
    clustered_data = []
    cluster_mean = []
    for i in range(cluster_num):
        clustered_data.append([])
    for idx,data in enumerate(data_set):
        cluster_idx = np.where(label_list == labels[idx])[0][0]
        clustered_data[cluster_idx].append(data)
        
    clustered_data = np.array(clustered_data)
    for i in range(cluster_num):
        cluster_mean.append(np.mean(clustered_data[0],axis=0))
    cluster_mean = np.array(cluster_mean)
    return clustered_data, cluster_mean



def knn(trains, tests,train_label, test_label):
    correct = 0
    for Did,data in enumerate(tests):
        print('ID: {}'.format(Did))
        distance = []
        for train in trains:
            dist = cdist(data.reshape((1,-1)), train.reshape((1,-1)))[0][0]
            distance.append(dist)
        idx = np.argsort(distance)
        neighbor_labels = train_label[idx]
        unique, pos = np.unique(neighbor_labels[0:KNN_K], return_inverse=True)
        counts = np.bincount(pos)
        maxpos = counts.argmax()
        predict_label = unique[maxpos]
        print('Predict label: {}'.format(predict_label))
        print('Actual label: {}'.format(test_label[Did]))
        print('')
        if(predict_label == test_label[Did]):
            correct += 1
    print('Accuracy: {}'.format(correct/len(tests)))



def visualization(eigenVec, reconstruct_face, method):
	#showing eigen and fisherfaces
	for i in range(25):
		plt.subplot(5, 5, i+1)
		plt.imshow(eigenVec[:,i].reshape(shape), cmap = 'gray')
		plt.savefig(method+'.png')

	#reconstruct faces
	idx = np.random.randint(train_size, size = 10)
	for i,id in enumerate(idx):
		plt.subplot(2, 5, i+1)
		plt.imshow(reconstruct_face[id].reshape(shape), cmap = 'gray')
		plt.savefig(method+'_'+str(idx)+'_'+'.png')



def PCA(train_data, train_label, test_data, test_label):
	print('PCA')
	mean_face = ((np.sum(train_data,axis = 0)/train_size).astype(int)).reshape((1,-1))
	Xc = train_data - mean_face
	COV_T = Xc @ Xc.T
	eigenVal, eigenVec = linalg.eigh(COV_T)
	idx = np.argsort(eigenVal)[::-1]
	eigenVal = eigenVal[idx]
	U = eigenVec[:,idx]
	W = (Xc.T @ U)[:,0:25]
	for i in range(25):
	    W[:,i] = W[:,i]/linalg.norm(W[:,i])

	proj = Xc @ W
	reconstruct_face = proj @ W.T + mean_face
	visualization(W, reconstruct_face, 'PCA')


	new_train = []
	new_test = []
	for i in range(train_size):
	    proj_coef = (train_data[i,:]- mean_face) @ W
	    new_train.append(proj_coef)

	for i in range(test_size):
	    proj_coef = (test_data[i,:]- mean_face) @ W
	    new_test.append(proj_coef)
	    
	new_train = np.array(new_train)
	new_test = np.array(new_test)
	knn(new_train, new_test, train_label, test_label)



def kernelPCA(train_data, train_label, test_data, test_label):
	print('kernelPCA')
	data = np.vstack((train_data, test_data))
	K = RBF_kernel(data)
	N1 = np.full((K.shape), 1/K.shape[0])
	Kc = K - N1 @ K - K @ N1 + N1 @ K @ N1
	eigenVal, eigenVec = linalg.eig(Kc)
	idx = np.argsort(eigenVal)[::-1]
	W = eigenVec[:,idx][:,:25].real
	for i in range(25):
		W[:,i] = W[:,i]/linalg.norm(W[:,i])
	proj = Kc.T @ W
	newtrain = proj[:train_size]
	newtest = proj[train_size:]
	knn(newtrain, newtest , train_label, test_label)



def LDA(train_data, train_label, test_data, test_label):
	print('LDA')
	mean_face = ((np.sum(train_data,axis = 0)/train_size).astype(int)).reshape((1,-1))
	Xc = train_data - mean_face
	mean = np.mean(train_data,axis=0)
	data, cluster_mean = clustering_by_label(train_data, train_label)

	#compute Sw, Sb
	Sw = np.zeros((shape[0]*shape[1],shape[0]*shape[1]))
	Sb = np.zeros((shape[0]*shape[1],shape[0]*shape[1]))
	for i in range(len(cluster_mean)):
	    data[i] = data[i] - cluster_mean[i]
	    for j in range(len(data[i])):
	        mat = data[i][j].reshape((-1,1)) @ data[i][j].reshape((1,-1))
	        Sw += mat
	cluster_mean = cluster_mean - mean
	for i in range(len(cluster_mean)):
	    ni = len(data[i])
	    mat = ni * cluster_mean[i].reshape((-1,1)) @ cluster_mean[i].reshape((1,-1))
	    Sb += mat


	#solve eigenvector problem
	eigenVal, eigenVec = linalg.eig(np.linalg.pinv(Sw) @ Sb)
	idx = np.argsort(eigenVal)[::-1]
	W = eigenVec[:,idx]
	W = W[:,:25].real
	for i in range(25):
	    W[:,i] = W[:,i]/linalg.norm(W[:,i])
	proj = Xc @ W
	reconstruct_face = proj @ W.T + mean_face
	visualization(W, reconstruct_face, 'LDA')

	new_train = []
	new_test = []
	for i in range(train_size):
	    proj_coef = (train_data[i,:]- mean_face) @ W
	    new_train.append(proj_coef)

	for i in range(test_size):
	    proj_coef = (test_data[i,:]- mean_face) @ W
	    new_test.append(proj_coef)
	    
	new_train = np.array(new_train)
	new_test = np.array(new_test)
	knn(new_train, new_test, train_label, test_label)



def kernelLDA(train_data, train_label, test_data, test_label):
	print('kernelLDA')
	data = np.vstack((train_data, test_data))
	data_label = np.hstack((train_label, test_label))
	K = RBF_kernel(data)
	mean = np.mean(K,axis=0)
	data, cluster_mean = clustering_by_label(train_data, train_label)
	M = np.zeros((total_size,total_size))
	N = np.zeros((total_size,total_size))
	clusters = np.unique(train_label)
	for cluster in clusters:
	    row_id = np.where(data_label == cluster)[0]
	    Ki = K[row_id, :]
	    l = len(row_id)
	    N1 = np.full((l,l), 1/l)
	    N = N + Ki.T @ (np.identity(l) - N1) @ Ki
	    cluster_mean = np.mean(Ki, axis=0)
	    M = M + l * (cluster_mean - mean).T @ (cluster_mean - mean)
	eigenVal, eigenVec = linalg.eig(np.linalg.pinv(N) @ M)
	idx = np.argsort(eigenVal)[::-1]
	W = eigenVec[:,idx][:,:25].real
	for i in range(25):
	    W[:,i] = W[:,i]/linalg.norm(W[:,i])
	proj = K.T @ eigenVec
	newtrain = proj[:train_size]
	newtest = proj[train_size:]
	knn(newtrain, newtest , train_label, test_label)



if __name__ == "__main__":
	train_data, train_label, test_data, test_label = dataLoader()
	PCA(train_data, train_label, test_data, test_label)
	kernelPCA(train_data, train_label, test_data, test_label)
	LDA(train_data, train_label, test_data, test_label)
	kernelLDA(train_data, train_label, test_data, test_label)