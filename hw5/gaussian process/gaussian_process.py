import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def data_processing(input_name):
	fp = open(input_name, 'r')
	data = []
	for line in fp:
		data.append(line.strip().split(' '))
	return np.array(data).astype(np.float)


def data_visualization(data):
    plt.scatter(data[:,0], data[:,1],color='red')


def kernel_function(X1,X2, amp, ls, alpha):   #compute kernel(gram) matrix, which use rational quadratic kernel as kernel function
    kernel = np.zeros((len(X1),len(X2)))
    for i in range(len(X1)):
        for j in range(len(X2)):
            value = amp**2 * (1+((X1[i]-X2[j])**2)/(2*alpha*ls**2))**(-alpha)
            kernel[i,j] = value
    return kernel


def gaussian_process(train_data, test_data, amp, ls, alpha, part):
    #compute gram matrix of training data and add noice observation
    #amp: overall variance
    #ls: lengthscale
    #alpha:  scale-mixture
    
    kernel = kernel_function(train_data[:,0],train_data[:,0],amp, ls, alpha)
    C = kernel + np.identity(train_size)*(1/beta)                               
    
    #predict step, use both training and testing data to compute kernel elements 
    #in new gram matrix(union gaussian of traing set and testing set) 
    kernel_test_train = kernel_function(train_data[:,0],test_data,amp, ls, alpha) 
    kernel_test_self = kernel_function(test_data,test_data,amp, ls, alpha)
    
    #use new gram matrix to compute mean vector and variance matrix of new gaussian distribution
    mean_vector = (kernel_test_train.T @ np.linalg.inv(C) @ (train_data[:,1]))
    kernel_newpoint = (kernel_test_self + np.identity(len(test_data))*(1/beta)) - (kernel_test_train.T @ np.linalg.inv(C) @ kernel_test_train) 
    
    #plotting results
    diff = 1.96 * np.sqrt(np.diag(kernel_newpoint))
    plt.plot(test_data,mean_vector)
    plt.fill_between(test_data[:,0], mean_vector+diff, mean_vector-diff, facecolor = "pink")
    plt.plot(test_data,mean_vector+diff, color = 'red')
    plt.plot(test_data,mean_vector-diff, color = 'red')
    plt.scatter(train_data[:,0], train_data[:,1],color='red',s=5)
    plt.savefig(part + '.png')
    plt.close()
    print('result parameter and results: ')
    print('amplitude: ', amp)
    print('lengthscale: ', ls)
    print('alpha: ', alpha)


def negative_log_likelihood_loss(para, train_data):
	amp = para[0]
	ls = para[1]
	alpha = para[2]
	kernel = kernel_function(train_data[:,0],train_data[:,0],amp , ls, alpha)
	C = kernel + np.identity(train_size)*(1/beta)
	loss_value = 0.5 * train_data[:,1].T @ np.linalg.inv(C) @ train_data[:,1] + 0.5 * len(train_data[:,1]) * np.log(2*np.pi) + 0.5 * np.log(np.linalg.det(C))
	return loss_value


if __name__== '__main__':
	


	input_data = 'input.txt'
	train_size = 34
	beta = 5


	#part 1
	print('Part 1:')
	amp = 3
	ls = 5
	alpha = 300
	data = data_processing(input_data)
	test_data = np.linspace(-60, 60, 1000).reshape(-1,1)
	gaussian_process(data,test_data, amp, ls, alpha, 'part1')  

	#part 2 optimize kernel parameter
	init_guess = [amp, ls, alpha]
	print('')
	print('Part 2:')
	print('Initial parameters: ')
	print('amplitude: ', amp)
	print('lengthscale: ', ls)
	print('alpha: ', alpha)
	print('')
	para = minimize(negative_log_likelihood_loss,init_guess,args=(data))
	gaussian_process(data,test_data, para.x[0], para.x[1], para.x[2], 'part2')

