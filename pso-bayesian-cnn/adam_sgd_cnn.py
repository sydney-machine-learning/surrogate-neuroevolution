import sklearn
import numpy as np
import time
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers import Activation

def load_dataset(size):
  (trainX, trainY), (testX, testY) = mnist.load_data()
  trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
  testX = testX.reshape((testX.shape[0], 28, 28, 1))
  trainY = to_categorical(trainY)
  testY = to_categorical(testY)
  trainX = trainX[:size]
  trainY = trainY[:size]
  testX = testX[:size]
  testY = testY[:size]
  return trainX, trainY, testX, testY

def define_model(optim = 'adam'):
  model = Sequential()
  model.add(Conv2D(32, (5, 5), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_uniform'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Flatten())
  model.add(Activation('relu'))
  model.add(Dense(10, activation='softmax'))
  if(optim =='sgd'):
    opt = SGD(lr=0.01)
  else:
    opt = Adam(learning_rate=0.001)  
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def main():
  outfile =open('ad_sgd.txt','a+')
  size = 10000
  trainX, trainY, testX, testY = load_dataset(size)
  trainX = (trainX.astype('float32'))/255.0
  testX = (testX.astype('float32'))/255.0
  time1 = time.time()
  model = define_model('sgd')
  model.fit(trainX, trainY, epochs=20, batch_size=200, verbose=2)
  _, train_perf = model.evaluate(trainX, trainY, verbose =2,batch_size = 200)
  _, test_perf = model.evaluate(testX, testY, verbose =2,batch_size = 200)
  time2 = time.time()
  time_f = (time2 - time1)/60
  print("train_perf:",train_perf,' ',"test_perf:",test_perf,' ',"time:",time_f)
  res = np.asarray([train_perf , test_perf , time_f])
  np.savetxt(outfile,  res   , fmt='%1.4f', newline='   ' )
  np.savetxt(outfile,  [' sgd'], fmt="%s", newline=' \n '  )

if __name__ == "__main__": main()
