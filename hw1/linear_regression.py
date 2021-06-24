import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




def data_processing(inputname):
	fp = open(inputname, 'r')
	data = []
	for line in fp:
		data.append(line.strip().split(","))
	data_np = np.array(data)
	data_np = data_np.astype(float)
	return data_np


def design_matrix(data):
    x_data = data[:,0]
    y_data = data[:,1]
    des_mat = []
    for i in range(len(x_data)):
        line = []
        for j in range(BASES):
            line.append(x_data[i]**j)
        des_mat.append(line)
    des_mat_np = np.array(des_mat)
    y_data = np.array(y_data)
    x_data = np.array(x_data)
    return des_mat_np, x_data, y_data

def LU_decomposition(mat):
    U = mat
    L = np.identity(BASES)
    for n in range(BASES):
        for m in range(n+1,BASES):
            L[m,n] = U[m,n]/U[n,n]
            U[m,:] = U[m,:]-U[n,:]*(U[m,n]/U[n,n])
    return L,U


def transpose(mat):
    len_n = len(mat[0,:])
    len_m = len(mat[:,0])
    new_mat = np.zeros((len_n,len_m))
    for j in range(len_n):
        for i in range(len_m):
            new_mat[j,i] = mat[i,j]
    return new_mat

def matrix_operation(des_mat):
    mat_ata = np.dot(transpose(des_mat),des_mat)
    if(LAMBDA!=0):
        mat_ata = mat_ata +LAMBDA*np.identity(BASES)
    return mat_ata


def inverse(mat):
    B = np.identity(BASES)
    for n in range(BASES):
        for m in range(n+1,BASES):
            a = mat[m,n]/mat[n,n]
            mat[m,:] = mat[m,:]-mat[n,:]*a
            B[m,:] = B[m,:] - B[n,:]*a
    for n in range(BASES-1,-1,-1):
        pivot = mat[n,n]
        mat[n,:] = mat[n,:]/pivot
        B[n,:] = B[n,:]/pivot
        for m in range(n-1,-1,-1):
            ele = mat[m,n]
            B[m,:] = B[m,:]-B[n,:]*ele
            mat[m,n] = 0
    return B


def visualization(coef, data, method):

	plt.scatter(data[:,0], data[:,1],color='red')
	t = np.linspace(-6, 6, 10000)
	y = 0
	for i in range(BASES):
	    y +=  coef[i]*(t**i)
	plt.plot(t,y)
	plt.savefig(method+'.png')
	plt.close()

	error = 0
	for i in range(len(data)):
	    ans = 0
	    for j in range(BASES):
	        ans += coef[j]*(data[i,0]**j)
	    error += (ans-data[i,1])**2

	result = ""
	for i in range(BASES):
	    result += str(coef[i])
	    result += "X^"
	    result += str(i)
	    if(i!=BASES-1):
	        result += "  +  "
	print('')
	print(method + ": ")
	print("Fitting line:")
	print(result)
	print("Total Error: ",(error))
	print('')



def LSE(data):
	des_mat, x_data, y_data = design_matrix(data)
	M = matrix_operation(des_mat)
	L, U = LU_decomposition(M)
	a = np.dot(inverse(U), inverse(L))
	a = np.dot(a, des_mat.T)
	coef = a @ y_data
	visualization(coef, data, 'LSE')



def Newton(data):
	des_mat, x_data, y_data = design_matrix(data)
	ATA = np.dot(transpose(des_mat), des_mat)
	ATB = np.dot(transpose(des_mat), y_data)
	newton_coef = np.dot(inverse(ATA),ATB)
	visualization(newton_coef, data, 'Newton')



if __name__ == "__main__":

	BASES = 3
	LAMBDA = 0
	inputfile = 'input.txt'

	data = data_processing(inputfile)
	LSE(data)
	Newton(data)