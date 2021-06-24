import numpy as np
import math


def uni_gaussian(mean,var):
	sample_total = np.sum(np.random.uniform(0,1,12))-6
	return mean + (var**0.5)*sample_total


if __name__== '__main__':
	mean = 3.0
	var = 5.0
	mean_old = mean
	var_old = var
	mean_n = 0
	var_n = 0
	n = 0
	print('Data point source function: N(',mean,',',var,')')

	while(True):
		n+=1
		data_n = uni_gaussian(mean,var)
		print('')
		print('Add data point:', data_n)
		mean_n = mean_old +(data_n-mean_old)/n
		var_n = (var_old + (data_n-mean_old)*(data_n-mean_n))
		print('Mean = ', mean_n, 'Variance = ', var_n/n)
		print('')
		if( abs(mean_n - mean_old) < 0.0001 and abs(var_n/n - var_old/(n-1))<0.0001):
			break
		mean_old = mean_n
		var_old = var_n