 

import random
import math  
import numpy as np 


bits = 6

data = np.genfromtxt('input/data'+str(bits)+'bits.txt',delimiter=',') # change this as needed

data_mask = np.zeros((data.shape[0], data.shape[1]+2))


for i in range(data.shape[0]):
  line = data[i,:]
  count= np.count_nonzero(line == 1)
  if count%2 ==0: 
    x = np.concatenate((line, np.array([1, 0]))) 
    data_mask[i,:] = x
  else: 
    x = np.concatenate((line, np.array([0, 1]))) 
    data_mask[i,:] = x

print(data_mask)

np.savetxt('data'+str(bits)+'bits_.txt', data_mask, fmt='%1.0f')

