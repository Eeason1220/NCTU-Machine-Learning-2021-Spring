import numpy as np
import matplotlib.pyplot as plt
from libsvm.svmutil import *



def data_preprocessing():
	x_test = np.loadtxt(open("X_test.csv", "rb"), delimiter=",")
	x_train = np.loadtxt(open("X_train.csv", "rb"), delimiter=",")
	y_test = np.loadtxt(open("Y_test.csv", "rb"), delimiter=",")
	y_train = np.loadtxt(open("Y_train.csv", "rb"), delimiter=",")
	return x_train, x_test, y_train, y_test




def Part1_kernel_compare():
    
    #Applying different kernels.
    #The parameter format is based on libsvm's document
    #function usage:
    #    svm_train(labels, features, options)
    #    svm_test(labels_test, features_test, trained model)
        
    #    For model parameters, cost c use default value:1
    #    As for kernel parameters, see the following command.
    
    
    
    #    linear kernel:
    #   -t 0: linear kernel mode
    
    #linear kernel
    print('Linear kernel:')
    print('cost = 1')
    
    model = svm_train(y_train, x_train, '-t 0')
    res = svm_predict(y_test, x_test, model)
    
    
    '''
        polynomial kernel:
        -t 1:: polynomial kernel mode
        -d 2:: degree of polynomial kernel: 2
        -r 10:: constant coef: 10
    '''
    #polynomial kernel
    print('----------')
    print('Polynomial kernel:')
    print('cost = 1')
    print('degree = 2')
    print('coef0 = 10')
    model = svm_train(y_train, x_train, '-t 1 -d 2 -r 10')
    res = svm_predict(y_test, x_test, model)
    
    
    '''
        RBF kernel:
        -t 2:: RBF kernel mode
        gamma:1/(#number of features)
    '''
    #RBF kernel
    print('----------')
    print('RBF kernel:')
    print('gamma = 1/784')
    model = svm_train(y_train, x_train, '-t 2')
    res = svm_predict(y_test, x_test, model)


def tuning(arg, arg_opt,acc_max):
    #to compare the local maximum accurate and turing parameter
    acc = svm_train(y_train, x_train, arg)
    if (acc_max<acc):
        acc_max = acc
        arg_opt = arg
    return acc_max, arg_opt


def Part2_grid_search():
    for i in range(3):
        if (i==0): #linear kernel
            arg_opt = ''
            acc_max = 0.0
            print("parameter bound for Linear Kernel: ")
            print('C = [0.001,0.01,0.1,1,10,100]')
            print('')
            for C in [0.001,0.01,0.1,1,10,100]: 
                arg = "-t %d -c %f -v 4" % (i, C)
                print('C =', C)
                acc_max, arg_opt = tuning(arg, arg_opt,acc_max)
            print('optimal option:', arg_opt) 	
            print('-----------')
        '''if (i==1): #polynomial kernel
            arg_opt = ''
            acc_max = 0.0
            for C in [0.001,0.01,0.1,1,10,100]:
                for gamma in [0.001,0.01,0.1,1,10,100]:
                    for coef in range(11):
                        for degree in range(1,4):
                            arg = "-t %d -d %d -g %f -r %d -c %f -v 4" % (i,degree,gamma,coef,C)
                            print(arg)
                            acc_opt, arg_opt = tuning(arg, arg_opt,acc_max)
            print('optimal parameter: ',arg_opt)'''
        
        if (i==2): #RBF kernel
            arg_opt = ''
            acc_max = 0.0
            print("parameter bound for RBF Kernel: ")
            print('C = [0.001,0.01,0.1,1,10,100]')
            print('gamma = [0.001,0.0015,0.002,0.0025,0.003]')
            print('')
            for C in [0.001,0.01,0.1,1,10,100]:
                for gamma in [0.001,0.0015,0.002,0.0025,0.003]:
                    arg = "-t %d -g %f -c %f -v 4" % (i,gamma,C)
                    print('C =', C, 'gamma =', gamma)
                    acc_opt, arg_opt = tuning(arg, arg_opt,acc_max)
            print('optimal option: ',arg_opt)


def linear_kernel(X1, X2):
    return X1.dot(X2.transpose())


def RBF_kernel(X1, X2, gamma):
    kernel = np.zeros((len(X1[:,0]),len(X2[:,0])))
    for i in range(len(X1[:,0])):
        for j in range(len(X2[:,0])):
            dis = np.sum((X1[i,:]-X2[j,:])**2)
            value = np.exp(-gamma * dis)
            kernel[i,j] = value
    return kernel


def Part3_user_defined_kernel():
    
    #The usage of precomputed kernel is shown in README of libsvm:
    #https://github.com/cjlin1/libsvm/blob/master/README

    '''
    First, we use kernel function, training data and testing data to compute the kernel matrix.
    '''
    linear_train = linear_kernel(x_train, x_train)
    RBF_train = RBF_kernel(x_train, x_train, 1/784)
    linear_test = linear_kernel(x_train, x_test).transpose()
    RBF_test = RBF_kernel(x_train, x_test, 1/784).transpose()
    
    '''
    Add both result in linearkernel and RBF kernel,
    According to README in libsvm, we need to add "ID" column in the head of training data
    for testing data, we can just add a column wiht arbitrary numbers in the head of data
    '''
    train_feature = linear_train + RBF_train
    test_feature = linear_test + RBF_test
    train_feature = np.hstack((np.arange(1, 5001).reshape((-1, 1)), train_feature))
    test_feature = np.hstack((np.arange(1, 2501).reshape((-1, 1)), test_feature))
    
    '''
    '-t 4' means mode 4:precomputed kernel.
    use the model to do the predition
    '''
    model = svm_train(y_train, train_feature, '-t 4')
    svm_predict(y_test, test_feature, model)


if __name__== '__main__':

	x_train, x_test, y_train, y_test = data_preprocessing()
	print('Part1_kernel_compare:')
	print('')
	Part1_kernel_compare()
	print('Part2_grid_search')
	print('')
	Part2_grid_search()
	print('Part3_user_defined_kernel')
	print('')
	Part3_user_defined_kernel()	
