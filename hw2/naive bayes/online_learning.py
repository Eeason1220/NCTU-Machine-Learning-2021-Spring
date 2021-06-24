import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys



def data_processing(inputname):
	fp = open(inputname, 'r')
	data = []
	for line in fp:
		data.append(line.strip())
	return data


def fact(n):
	if (n==1 or n==0): return 1
	else: return n*fact(n-1)


def Comb(n,m):
	t = max(m,n-m)
	mult = 1
	for k in range(n,t,-1):
		mult = mult*k
	div = fact(n-t)
	return mult/div


def likelihood_function(n,m, idx,  data_length):
	coef = Comb(n,m)
	t = np.linspace(0, 1, 10000)
	y = 0
	y = coef * (t**m * (1-t)**(n-m))  #Binomial distribution
	#fig,ax = plt.subplots()
	plt.subplot(data_length, 3, 3*idx+2)
	plt.plot(t, y)
	plt.title('Likelyhood'+str(idx+1))
	#ax.set_title('Likelyhood'+str(idx+1))

def gamma(m):
	if (m==1 or m==2): return 1
	else:
		return fact(m-1)


def beta_plot(a,b, mode, idx, data_length):
	t = np.linspace(0, 1, 10000)
	coef = gamma(a+b)/(gamma(a)*gamma(b)) #beta function
	y = 0
	y = (t**(a-1) * (1-t)**(b-1)*coef)
	#fig,ax = plt.subplots()
	if(mode == 'prior'):
		plt.subplot(data_length, 3, 3*idx+1)
		plt.plot(t, y)
		plt.title('prior'+str(idx+1))
		#ax.set_title('prior'+str(idx+1))
	else:
		plt.subplot(data_length, 3, 3*idx+3)
		plt.plot(t, y)
		plt.title('prior'+str(idx+1))
		#ax.set_title('posterior'+str(idx+1))



if __name__ == "__main__":

	inputfile = 'input.txt'
	data = data_processing(inputfile)
	A = 1
	B = 1
	plt.figure(figsize=(12,16))
	for i in range(len(data)):
		line = data[i]
		cnt = 0
		N = len(line)
		prior_A = A
		prior_B = B
		print("case",i+1,": ",line)
		for j in range(len(line)):
			if line[j] == '1':
				cnt += 1
		MLE = (cnt)/(N)
		likelihood = Comb(N,cnt)*(MLE**cnt) *((1-MLE)**(N-cnt))
		A = A + cnt
		B = B + (N - cnt)
		print("Likelihood: ",likelihood)
		print("Beta prior: a = ", prior_A,", b = ", prior_B )
		print("Beta posterior: a = ", A,", b = ", B )
		beta_plot(prior_A,prior_B, 'prior', i, len(data))
		likelihood_function(N, cnt, i, len(data))
		beta_plot(A, B, 'posterior', i, len(data))
		print("")
	plt.tight_layout()
	plt.savefig('result.png')
