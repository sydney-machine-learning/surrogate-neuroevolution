import sklearn
import time
import sys
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		end_ix = i + n_steps
		if end_ix > len(sequences)-1:
			break
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def grad(x_train , y_train , x_test , y_test,n_steps ,opt = 'adam'):
  n_features = x_train.shape[-1]
  n_t = x_train.shape[0]
  x_train = x_train.reshape(n_t , n_steps,n_features,1)
  #x_train = x_train.reshape(n_t , 1,n_steps,n_features)
  model = Sequential()
  model.add(Conv2D(filters=6, kernel_size=(2,3), activation='relu', input_shape=(n_steps, n_features,1)))
  model.add(MaxPooling2D(pool_size=(2,2),padding = "same"))
  model.add(Conv2D(filters=16, kernel_size=3, activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2),padding = "same"))
  model.add(Flatten())
  model.add(Activation('relu'))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(n_features))
  model.compile(optimizer= opt, loss='mse')
  #model.fit(x_train, y_train, batch_size = 16,epochs=2000, verbose=2)
  model.fit(x_train, y_train,epochs=500, verbose=2)

  rmse_train = 0
  for i in range(x_train.shape[0]):
    y_hat_train = model.predict(x_train[i].reshape(1,n_steps,n_features,1) , verbose = 0)
    rmse_train += ((y_train[i] - y_hat_train)**2).sum()

  rmse_test = 0
  for i in range(x_test.shape[0]):
    y_hat_test = model.predict(x_test[i].reshape(1,n_steps,n_features,1) , verbose = 0)
    rmse_test += ((y_test[i] - y_hat_test)**2).sum()
    
  rmse_train = np.sqrt(rmse_train/len(x_train))  
  rmse_test = np.sqrt(rmse_test/len(x_test))
  return rmse_train,rmse_test  



def main():
  problem = "Time-Series"
  step_size = int(sys.argv[1])
  opti = int(sys.argv[2])
  #path_results = 'results/' + problem +'/results_new.txt'
  #outfile_pso=open(path_results,'a+')
  path_temp = 'results/' + problem +'/temp.txt'
  outfile_temp = open(path_temp,'a+')
  name = problem
  data = np.genfromtxt('DATA/time-series/ashok_mar19_mar20.csv', delimiter=',')[1:,1:]
  data = preprocessing.normalize(data)
  X,y = split_sequences(data,step_size)
  X, y = shuffle_in_unison(X, y)
  train_size = 304
  x_train = np.expand_dims(X[:train_size, :, :], axis=1)
  y_train = y[:train_size, :]
  x_test = np.expand_dims(X[train_size:, :, :], axis=1)
  y_test = y[train_size:, :]
  print("x_train:" , x_train.shape)
  print("y_train:" , y_train.shape)
  print("x_test:" , x_test.shape)
  print("y_test:" , y_test.shape)
  if opti == 0:
    optim = 'adam'
  else:
    optim = 'sgd'  
  time1 = time.time()
  train_rmse,test_rmse = grad(x_train , y_train , x_test , y_test,step_size ,optim)
  time2 = time.time()
  time_f = (time2-time1)/60
  print("train_rmse:",train_rmse,' ',"test_rmse:",test_rmse,' ',"time:",time_f)
  res = np.asarray([train_rmse , test_rmse , time_f])
  np.savetxt(outfile_temp,  res   , fmt='%1.4f', newline='   ' )
  np.savetxt(outfile_temp,  [' Benchmark: ' + optim], fmt="%s", newline=' \n '  )



if __name__ == "__main__": main()