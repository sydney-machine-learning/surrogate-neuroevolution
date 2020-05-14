# !/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
import random
import time
import math
import os
import shutil

import multiprocessing 
import gc
  
import copy    # array-copying convenience
import sys     # max float

# -----------------------------------

# using https://github.com/sydney-machine-learning/canonical_neuroevolution



class neuralnetwork:

	def __init__(self, Topo, Train, Test, learn_rate):
		self.Top = Topo  # NN topology [input, hidden, output]
		self.TrainData = Train
		self.TestData = Test
		self.lrate = learn_rate
		self.W1 = np.random.randn(self.Top[0], self.Top[1]) / np.sqrt(self.Top[0])
		self.B1 = np.random.randn(1, self.Top[1]) / np.sqrt(self.Top[1])  # bias first layer
		self.W2 = np.random.randn(self.Top[1], self.Top[2]) / np.sqrt(self.Top[1])
		self.B2 = np.random.randn(1, self.Top[2]) / np.sqrt(self.Top[1])  # bias second layer
		self.hidout = np.zeros((1, self.Top[1]))  # output of first hidden layer
		self.out = np.zeros((1, self.Top[2]))  # output last layer
		self.pred_class = 0
 

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def sampleEr(self, actualout):
		error = np.subtract(self.out, actualout)
		sqerror = np.sum(np.square(error)) / self.Top[2]
		return sqerror

	def ForwardPass(self, X):
		z1 = X.dot(self.W1) - self.B1
		self.hidout = self.sigmoid(z1)  # output of first hidden layer
		z2 = self.hidout.dot(self.W2) - self.B2
		self.out = self.sigmoid(z2)  # output second hidden layer

		self.pred_class = np.argmax(self.out)


		## print(self.pred_class, self.out, '  ---------------- out ')

	'''def BackwardPass(self, Input, desired):
		out_delta = (desired - self.out).dot(self.out.dot(1 - self.out))
		hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))
		# print(self.B2.shape)
		self.W2 += (self.hidout.T.reshape(self.Top[1],1).dot(out_delta) * self.lrate)
		self.B2 += (-1 * self.lrate * out_delta)
		self.W1 += (Input.T.reshape(self.Top[0],1).dot(hid_delta) * self.lrate)
		self.B1 += (-1 * self.lrate * hid_delta)'''




	def BackwardPass(self, Input, desired): # since data outputs and number of output neuons have different orgnisation
		onehot = np.zeros((desired.size, self.Top[2]))
		onehot[np.arange(desired.size),int(desired)] = 1
		desired = onehot
		out_delta = (desired - self.out)*(self.out*(1 - self.out))
		hid_delta = np.dot(out_delta,self.W2.T) * (self.hidout * (1 - self.hidout))
		self.W2 += np.dot(self.hidout.T,(out_delta * self.lrate))
		self.B2 += (-1 * self.lrate * out_delta)
		Input = Input.reshape(1,self.Top[0])
		self.W1 += np.dot(Input.T,(hid_delta * self.lrate))
		self.B1 += (-1 * self.lrate * hid_delta)


	def decode(self, w):
		w_layer1size = self.Top[0] * self.Top[1]
		w_layer2size = self.Top[1] * self.Top[2]

		w_layer1 = w[0:w_layer1size]
		self.W1 = np.reshape(w_layer1, (self.Top[0], self.Top[1]))

		w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
		self.W2 = np.reshape(w_layer2, (self.Top[1], self.Top[2]))
		self.B1 = w[w_layer1size + w_layer2size:w_layer1size + w_layer2size + self.Top[1]].reshape(1,self.Top[1])
		self.B2 = w[w_layer1size + w_layer2size + self.Top[1]:w_layer1size + w_layer2size + self.Top[1] + self.Top[2]].reshape(1,self.Top[2])



	def encode(self):
		w1 = self.W1.ravel()
		w1 = w1.reshape(1,w1.shape[0])
		w2 = self.W2.ravel()
		w2 = w2.reshape(1,w2.shape[0])
		w = np.concatenate([w1.T, w2.T, self.B1.T, self.B2.T])
		w = w.reshape(-1)
		return w

	def softmax(self):
		prob = np.exp(self.out)/np.sum(np.exp(self.out))
		return prob



	def langevin_gradient(self, data, w, depth):  # BP with SGD (Stocastic BP)

		self.decode(w)  # method to decode w into W1, W2, B1, B2.
		size = data.shape[0]

		Input = np.zeros((1, self.Top[0]))  # temp hold input
		Desired = np.zeros((1, self.Top[2]))
		fx = np.zeros(size)

		for i in range(0, depth):
			for i in range(0, size):
				pat = i
				Input = data[pat, 0:self.Top[0]]
				Desired = data[pat, self.Top[0]:]
				self.ForwardPass(Input)
				self.BackwardPass(Input, Desired)
		w_updated = self.encode()

		return  w_updated

	def evaluate_proposal(self, data, w ):  # BP with SGD (Stocastic BP)

		self.decode(w)  # method to decode w into W1, W2, B1, B2.
		size = data.shape[0]

		Input = np.zeros((1, self.Top[0]))  # temp hold input
		Desired = np.zeros((1, self.Top[2]))
		fx = np.zeros(size)
		prob = np.zeros((size,self.Top[2]))

		for i in range(0, size):  # to see what fx is produced by your current weight update
			Input = data[i, 0:self.Top[0]]
			self.ForwardPass(Input)
			fx[i] = self.pred_class
			prob[i] = self.softmax()

		## print(fx, 'fx')
		## print(prob, 'prob' )

		return fx, prob






















class evaluate_neuralnetwork(object):  # class for fitness func 
	def __init__(self,  netw, traindata, testdata):


		learn_rate = 0.1 # in case you wish to use gradients to help evolution
		self.neural_net = neuralnetwork(netw, traindata, testdata, learn_rate)  # FNN model, but can be extended to RNN model
		self.traindata = traindata
		self.testdata = testdata
		self.topology = netw
  


	def rmse(self, pred, actual):
		return np.sqrt(((pred-actual)**2).mean())

	def accuracy(self,pred,actual ):
		count = 0
		for i in range(pred.shape[0]):
			if pred[i] == actual[i]:
				count+=1
		return 100*(count/pred.shape[0])

	def classification_perf(self, x, type_data):

		if type_data == 'train':
			data = self.traindata
		else:
			data = self.testdata

		y = data[:, self.topology[0]]
		fx, prob = self.neural_net.evaluate_proposal(data,x)
		fit= self.rmse(fx,y) 
		acc = self.accuracy(fx,y) 

		return acc, fit

		
	def fit_func(self, x):    #  function  (can be any other function, model or diff neural network models (FNN or RNN ))
		  
		y = self.traindata[:, self.topology[0]]
		fx, prob = self.neural_net.evaluate_proposal(self.traindata,x)
		fit= self.rmse(fx,y) 


		return fit # note we will maximize fitness, hence minimize error



	def neuro_gradient(self, data, w, depth):  # BP with SGD (Stocastic BP)

		gradients = self.neural_net.langevin_gradient(data, w, depth)

		return gradients







 

class particle(evaluate_neuralnetwork):
	def __init__(self,  dim,  maxx, minx, netw, traindata, testdata):
		evaluate_neuralnetwork.__init__( self, netw, traindata, testdata) # inherits neuroevolution class definition and methods

		self.position = ((maxx - minx) * np.random.rand(dim)  + minx)
		self.velocity = ((maxx - minx) * np.random.rand(dim)  + minx)

		self.error =  self.fit_func(self.position) # curr error
		self.best_part_pos =  self.position.copy()
		self.best_part_err = self.error # best error 



class neuroevolution(evaluate_neuralnetwork, multiprocessing.Process):  # PSO http://www.scholarpedia.org/article/Particle_swarm_optimization
	def __init__(self, pop_size, dimen, max_evals,  max_limits, min_limits, netw, traindata, testdata, parameter_queue, wait_chain, event, island_id, swap_interval):
		
		multiprocessing.Process.__init__(self) # set up multiprocessing class

		evaluate_neuralnetwork.__init__( self, netw, traindata, testdata) # sepossiesiont up - inherits neuroevolution class definition and methods

		self.parameter_queue = parameter_queue
		self.signal_main = wait_chain
		self.event =  event


		self.island_id = island_id



		self.dim = dimen
		self.n = pop_size
		self.minx = min_limits
		self.maxx = max_limits
		self.max_evals = max_evals

		self.netw = netw
		self.traindata = traindata
		self.testdata = testdata

		self.swap_interval = swap_interval

		print('evoPSO initialized', island_id)



	def run(self): # this is executed without even calling - due to multi-processing


		rnd = random.Random(0)
		# create n random particles 

		swarm = [particle(self.dim, self.minx, self.maxx,  self.netw, self.traindata, self.testdata) for i in range(self.n)] 
	 
		best_swarm_pos = [0.0 for i in range(self.dim)] # not necess.
		best_swarm_err = sys.float_info.max # swarm best


	
		for i in range(self.n): # check each particle
 
			if swarm[i].error < best_swarm_err:
				best_swarm_err = swarm[i].error
				best_swarm_pos = copy.copy(swarm[i].position) 
			
		epoch = 0
		evals = 0
		w = 0.729    # inertia
		c1 = 1.49445 # cognitive (particle)
		c2 = 1.49445 # social (swarm)

		gradient_prob =0.5

		depth = 1 # num of epochs for gradients by backprop

		use_gradients = True


		


		self.event.clear()




		while evals < self.max_evals:

			
			for i in range(self.n): # process each particle 
				r1 = np.random.rand(self.dim)
				r2 = np.random.rand(self.dim)

				swarm[i].velocity = ( (w * swarm[i].velocity) + (c1 * r1 * (swarm[i].best_part_pos - swarm[i].position)) +  (c2 * r2 * (best_swarm_pos - swarm[i].position)) )  

				for k in range(self.dim): 
					if swarm[i].velocity[k] < self.minx[k]:
						swarm[i].velocity[k] = self.minx[k]
					elif swarm[i].velocity[k] > self.maxx[k]:
						swarm[i].velocity[k] = self.maxx[k]
 
				swarm[i].position += swarm[i].velocity

				u = random.uniform(0, 1)

				if u < gradient_prob and use_gradients == True: 

					swarm[i].position = self.neuro_gradient(self.traindata, swarm[i].position.copy(), depth)  
					
 
				swarm[i].error = self.fit_func(swarm[i].position)

				if swarm[i].error < swarm[i].best_part_err:
					swarm[i].best_part_err = swarm[i].error
					swarm[i].best_part_pos = copy.copy(swarm[i].position)

				if swarm[i].error < best_swarm_err:
					best_swarm_err = swarm[i].error
					best_swarm_pos = copy.copy(swarm[i].position)


				#print(' **  ', i, evals, epoch, best_swarm_err, self.island_id)

			if evals % (self.n*5)  == 0: 

				train_per, rmse_train = self.classification_perf(best_swarm_pos, 'train')
				test_per, rmse_test = self.classification_perf(best_swarm_pos, 'test')

				print(evals, epoch, train_per , rmse_train,  'classification_perf RMSE train * pso' )   
				#print(evals, epoch, test_per ,  rmse_test, 'classification_perf  RMSE test * pso' )



			if (evals % self.swap_interval == 0 ): # interprocess (island) communication for exchange of neighbouring best_swarm_pos
				param = best_swarm_pos
				self.parameter_queue.put(param)
				self.signal_main.set()
				self.event.clear()
				self.event.wait()
				result =  self.parameter_queue.get()
				best_swarm_pos = result 




			epoch += 1
			evals += self.n
 

		train_per, rmse_train = self.classification_perf(best_swarm_pos, 'train')
		test_per, rmse_test = self.classification_perf(best_swarm_pos, 'test')


		print(evals, epoch, train_per , rmse_train,  'classification_perf RMSE train * pso' )   
		print(evals, epoch, test_per ,  rmse_test, 'classification_perf  RMSE test * pso' )



		#return train_per, test_per, rmse_train, rmse_test

class distributed_neuroevo:

	def __init__(self,  pop_size, dimen, max_evals,  max_limits, min_limits, netw, traindata, testdata, num_islands):
		#FNN Chain variables
		self.traindata = traindata
		self.testdata = testdata
		self.topology = netw 
		self.pop_size = pop_size
		self.num_param =  dimen
		self.max_evals = max_evals
		self.max_limits = max_limits
		self.min_limits = min_limits

		self.num_islands = num_islands

		self.islands = [] 
		self.island_numevals = int(self.max_evals/self.num_islands) 

		# create queues for transfer of parameters between process islands running in parallel 
		self.parameter_queue = [multiprocessing.Queue() for i in range(num_islands)]
		self.island_queue = multiprocessing.JoinableQueue()	
		self.wait_island = [multiprocessing.Event() for i in range (self.num_islands)]
		self.event = [multiprocessing.Event() for i in range (self.num_islands)]


		self.swap_interval = pop_size


	def initialize_islands(self ):
		
		
		for i in range(0, self.num_islands):
			print(i, '   done initialize_islands')
 
			self.islands.append(neuroevolution(  self.pop_size, self.num_param, self.island_numevals,  self.max_limits, self.min_limits, self.topology, self.traindata, self.testdata ,self.parameter_queue[i],self.wait_island[i],self.event[i], i, self.swap_interval))
	
	def swap_procedure(self, parameter_queue_1, parameter_queue_2): 


			param1 = parameter_queue_1.get()
			param2 = parameter_queue_2.get() 

			swap_proposal = 0.5

			u = np.random.uniform(0,1)

			swapped = False
			if u < swap_proposal:   
				param_temp =  param1
				param1 = param2
				param2 = param_temp
				swapped = True 
			else:
				swapped = False 
			return param1, param2 ,swapped
 
 
	def evolve_islands(self): 
		# only adjacent chains can be swapped therefore, the number of proposals is ONE less islands

		self.initialize_islands()


		swap_proposal = np.ones(self.num_islands-1)
 
		# create parameter holders for paramaters that will be swapped
		replica_param = np.zeros((self.num_islands, self.num_param))  
		lhood = np.zeros(self.num_islands)
		# Define the starting and ending of MCMC Chains
		start = 0
		end = self.island_numevals
		number_exchange = np.zeros(self.num_islands) 

		#RUN MCMC CHAINS
		for l in range(0,self.num_islands):
			self.islands[l].start_chain = start
			self.islands[l].end = end 

		for j in range(0,self.num_islands):        
			self.wait_island[j].clear()
			self.event[j].clear()
			self.islands[j].start()
		#SWAP PROCEDURE

		swaps_appected_main =0
		total_swaps_main =0
		for i in range(int(self.island_numevals/self.swap_interval)):
			count = 0
			for index in range(self.num_islands):
				if not self.islands[index].is_alive():
					count+=1
					self.wait_island[index].set() 

			if count == self.num_islands:
				break 

			timeout_count = 0
			for index in range(0,self.num_islands): 
				flag = self.wait_island[index].wait()
				if flag: 
					timeout_count += 1

			if timeout_count != self.num_islands: 
				continue 

			for index in range(0,self.num_islands-1): 
				param_1, param_2, swapped = self.swap_procedure(self.parameter_queue[index],self.parameter_queue[index+1])
				self.parameter_queue[index].put(param_1)
				self.parameter_queue[index+1].put(param_2)
				if index == 0:
					if swapped:
						swaps_appected_main += 1
					total_swaps_main += 1
			for index in range (self.num_islands):
					self.event[index].set()
					self.wait_island[index].clear() 

		for index in range(0,self.num_islands):
			self.islands[index].join()
		self.island_queue.join()
		  

		#return   train_per, test_per, rmse_train, rmse_test



 


def main():


	#problem = 8

	method = 'pso'    # or 'rcga'

	for problem in range(5, 9) : 


		separate_flag = False # dont change 


		if problem == 1: #Wine Quality White
			data  = np.genfromtxt('DATA/winequality-red.csv',delimiter=';')
			data = data[1:,:] #remove Labels
			classes = data[:,11].reshape(data.shape[0],1)
			features = data[:,0:11]
			separate_flag = True
			name = "winequality-red"
			hidden = 50
			ip = 11 #input
			output = 10 
		if problem == 3: #IRIS
			data  = np.genfromtxt('DATA/iris.csv',delimiter=';')
			classes = data[:,4].reshape(data.shape[0],1)-1
			features = data[:,0:4]

			separate_flag = True
			name = "iris"
			hidden = 8  #12
			ip = 4 #input
			output = 3 
			#NumSample = 50000
		if problem == 2: #Wine Quality White
			data  = np.genfromtxt('DATA/winequality-white.csv',delimiter=';')
			data = data[1:,:] #remove Labels
			classes = data[:,11].reshape(data.shape[0],1)
			features = data[:,0:11]
			separate_flag = True
			name = "winequality-white"
			hidden = 50
			ip = 11 #input
			output = 10 
			#NumSample = 50000
		if problem == 4: #Ionosphere
			traindata = np.genfromtxt('DATA/Ions/Ions/ftrain.csv',delimiter=',')[:,:-1]
			testdata = np.genfromtxt('DATA/Ions/Ions/ftest.csv',delimiter=',')[:,:-1]
			name = "Ionosphere"
			hidden = 15 #50
			ip = 34 #input
			output = 2 

			#NumSample = 50000
		if problem == 5: #Cancer
			traindata = np.genfromtxt('DATA/Cancer/ftrain.txt',delimiter=' ')[:,:-1]
			testdata = np.genfromtxt('DATA/Cancer/ftest.txt',delimiter=' ')[:,:-1]
			name = "Cancer"
			hidden = 8 # 12
			ip = 9 #input
			output = 2 
			#NumSample =  50000

			# print(' cancer')

		if problem == 6: #Bank additional
			data = np.genfromtxt('DATA/Bank/bank-processed.csv',delimiter=';')
			classes = data[:,20].reshape(data.shape[0],1)
			features = data[:,0:20]
			separate_flag = True
			name = "bank-additional"
			hidden = 50
			ip = 20 #input
			output = 2 
			#NumSample = 50000
		if problem == 7: #PenDigit
			traindata = np.genfromtxt('DATA/PenDigit/train.csv',delimiter=',')
			testdata = np.genfromtxt('DATA/PenDigit/test.csv',delimiter=',')
			name = "PenDigit"
			for k in range(16):
				mean_train = np.mean(traindata[:,k])
				dev_train = np.std(traindata[:,k])
				traindata[:,k] = (traindata[:,k]-mean_train)/dev_train
				mean_test = np.mean(testdata[:,k])
				dev_test = np.std(testdata[:,k])
				testdata[:,k] = (testdata[:,k]-mean_test)/dev_test
			ip = 16
			hidden = 30
			output = 10 

			#NumSample = 50000
		if problem == 8: #Chess
			data  = np.genfromtxt('DATA/chess.csv',delimiter=';')
			classes = data[:,6].reshape(data.shape[0],1)
			features = data[:,0:6]
			separate_flag = True
			name = "chess"
			hidden = 25
			ip = 6 #input
			output = 18 


		
		#Separating data to train and test
		if separate_flag is True:
			#Normalizing Data
			for k in range(ip):
				mean = np.mean(features[:,k])
				dev = np.std(features[:,k])
				features[:,k] = (features[:,k]-mean)/dev
			train_ratio = 0.6 #Choosable
			indices = np.random.permutation(features.shape[0])
			traindata = np.hstack([features[indices[:np.int(train_ratio*features.shape[0])],:],classes[indices[:np.int(train_ratio*features.shape[0])],:]])
			testdata = np.hstack([features[indices[np.int(train_ratio*features.shape[0])]:,:],classes[indices[np.int(train_ratio*features.shape[0])]:,:]])



		topology = [ip, hidden, output] 

		netw = topology
		y_test =  testdata[:,netw[0]]
		y_train =  traindata[:,netw[0]]

 
		outfile_pso=open('resultspso.txt','a+')


		num_varibles = (netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] + netw[2]  # num of weights and bias
		max_limits = np.repeat(50, num_varibles) 
		min_limits = np.repeat(-50, num_varibles)



		for run in range(1, 2) :  

			max_evals = 20000
			pop_size =  100
			num_islands = 10

			timer = time.time()

			neuroevolution =  distributed_neuroevo(pop_size, num_varibles, max_evals,  max_limits, min_limits, netw, traindata, testdata, num_islands)


			neuroevolution.evolve_islands()

			#train_per, test_per, rmse_train, rmse_test = neuroevolution.evolve_islands()

			'''print(train_per , rmse_train,  'classification_perf RMSE train * pso' )   
			print(test_per ,  rmse_test, 'classification_perf  RMSE test * pso' )

			timer2 = time.time()
			timetotal = (timer2 - timer) /60


			allres =  np.asarray([ problem, run, train_per, test_per, rmse_train, rmse_test, timetotal]) 
			np.savetxt(outfile_pso,  allres   , fmt='%1.4f', newline='   '  )
			np.savetxt(outfile_pso,  ['  PSO'], fmt="%s", newline=' \n '  )'''


	 






	 


 














if __name__ == "__main__": main()
