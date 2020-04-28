# !/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
import random
import time
import math
import os
import shutil




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

		print( ' network set ')

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


class evolution(object):
	def __init__(self, pop_size, dimen, max_evals,  max_limits, min_limits, netw, traindata, testdata):


		learn_rate = 0.1 # in case you wish to use gradients to help evolution
		self.neural_net = neuralnetwork(netw, traindata, testdata, learn_rate)


		self.traindata = traindata
		self.testdata = testdata
		self.topology = netw


 
		# Evolution alg

		self.EPSILON = 1e-40  # convergence
		self.sigma_eta = 0.1
		self.sigma_zeta = 0.1
		self.children = 2
		self.num_parents = 3
		self.family = 2
		self.sp_size = self.children + self.family
		self.population =   np.random.randn( pop_size  , dimen)  * 5  #[SpeciesPopulation(dimen) for count in xrange(pop_size)]
		self.sub_pop =  np.random.randn( self.sp_size , dimen )  * 5  #[SpeciesPopulation(dimen) for count in xrange(NPSize)]
		self.fitness = np.random.randn( pop_size)
		self.sp_fit  = np.random.randn(self.sp_size)
		self.best_index = 0
		self.best_fit = 0
		self.worst_index = 0
		self.worst_fit = 0
		self.rand_parents =  self.num_parents
		self.temp_index =  np.arange(0, pop_size)
		self.rank =  np.arange(0, pop_size)
		self.list = np.arange(0, self.sp_size)
		self.parents = np.arange(0, pop_size)
		self.pop_size = pop_size
		self.dimen = dimen
		self.num_evals = 0
		self.max_evals = max_evals
		self.problem = 3


	def rmse(self, pred, actual):

		return np.sqrt(((pred-actual)**2).mean())

	def accuracy(self,pred,actual ):
		count = 0
		for i in range(pred.shape[0]):
			if pred[i] == actual[i]:
				count+=1
		return 100*(count/pred.shape[0])


 

		
	def fit_func(self, x):    #  function  (can be any other function, model or even a neural network)
		fit = 0.0
		if self.problem == 1: # rosenbrock
			for j in range(x.size -1):
				fit += (100.0*(x[j]*x[j] - x[j+1])*(x[j]*x[j] - x[j+1]) + (x[j]-1.0)*(x[j]-1.0))
		elif self.problem ==2:  # ellipsoidal - sphere function
			for j in range(x.size):
				fit = fit + ((j+1)*(x[j]*x[j]))

		elif self.problem ==3:

			y = self.traindata[:, self.topology[0]]
			fx, prob = self.neural_net.evaluate_proposal(self.traindata,x)
			fit= self.rmse(fx,y) 

			#print(fit, ' is fit')



		return fit # note we will maximize fitness, hence minimize error


	def rand_normal(self, mean, stddev):
		if (not evolution.n2_cached):
			#choose a point x,y in the unit circle uniformly at random
			x = np.random.uniform(-1,1,1)
			y = np.random.uniform(-1,1,1)
			r = x*x + y*y
			while (r == 0 or r > 1):
				x = np.random.uniform(-1,1,1)
				y = np.random.uniform(-1,1,1)
				r = x*x + y*y
			# Apply Box-Muller transform on x, y
			d = np.sqrt(-2.0*np.log(r)/r)
			n1 = x*d
			evolution.n2 = y*d
			# scale and translate to get desired mean and standard deviation
			result = n1*stddev + mean
			evolution.n2_cached = True
			return result
		else:
			evolution.n2_cached = False
			return evolution.n2*stddev + mean

	def evaluate(self):

 

		self.fitness[0] = self.fit_func(self.population[0,:])
		self.best_fit = self.fitness[0]
		for i in range(self.pop_size):
			self.fitness[i] = self.fit_func(self.population[i,:])
			if (self.best_fit> self.fitness[i]):
				self.best_fit =  self.fitness[i]
				self.best_index = i
		self.num_evals += 1

	# calculates the magnitude of a vector
	def mod(self, List):
		sum = 0
		for i in range(self.dimen):
			sum += (List[i] * List[i] )
		return np.sqrt(sum)

	def parent_centric_xover(self, current):
		centroid = np.zeros(self.dimen)
		tempar1 = np.zeros(self.dimen)
		tempar2 = np.zeros(self.dimen)
		temp_rand = np.zeros(self.dimen)
		d = np.zeros(self.dimen)
		D = np.zeros(self.num_parents)
		temp1, temp2, temp3 = (0,0,0)
		diff = np.zeros((self.num_parents, self.dimen)) 

 


		for i in range(self.dimen):
			for u in range(self.num_parents):
				
				centroid[i]  = centroid[i] +  self.population[self.temp_index[u],i]
		centroid   = centroid / self.num_parents 
		# calculate the distace (d) from centroid to the index parent self.temp_index[0]
		# also distance (diff) between index and other parents are computed
		for j in range(1, self.num_parents):
			for i in range(self.dimen):
				if j == 1:
					d[i]= centroid[i]  - self.population[self.temp_index[0],i]
				diff[j, i] = self.population[self.temp_index[j], i] - self.population[self.temp_index[0],i]
			if (self.mod(diff[j,:]) < self.EPSILON):
				print('Points are very close to each other. Quitting this run')
				return 0
		dist = self.mod(d)
		if (dist < self.EPSILON):
			print( " Error -  points are very close to each other. Quitting this run   ")
			return 0
		# orthogonal directions are computed
		for j in range(1, self.num_parents):
			temp1 = self.inner(diff[j,:] , d )
			if ((self.mod(diff[j,:]) * dist) == 0):
				print("Division by zero")
				temp2 = temp1 / (1)
			else:
				temp2 = temp1 / (self.mod(diff[j,:]) * dist)
			temp3 = 1.0 - np.power(temp2, 2)
			D[j] = self.mod(diff[j]) * np.sqrt(np.abs(temp3))
		D_not = 0.0
		for i in range(1, self.num_parents):
			D_not += D[i]
		D_not /= (self.num_parents - 1) # this is the average of the perpendicular distances from all other parents (minus the index parent) to the index vector
		evolution.n2 = 0.0
		evolution.n2_cached = False
		for i in range(self.dimen):
			tempar1[i] = self.rand_normal(0,  self.sigma_eta * D_not) #rand_normal(0, D_not * sigma_eta);
			tempar2[i] = tempar1[i]
		if(np.power(dist, 2) == 0):
			print(" division by zero: part 2")
			tempar2  = tempar1
		else:
			tempar2  = tempar1  - (    np.multiply(self.inner(tempar1, d) , d )  ) / np.power(dist, 2.0)
		tempar1 = tempar2
		self.sub_pop[current,:] = self.population[self.temp_index[0],:] + tempar1
		rand_var = self.rand_normal(0, self.sigma_zeta)
		for j in range(self.dimen):
			temp_rand[j] =  rand_var
		self.sub_pop[current,:] += np.multiply(temp_rand ,  d )
		self.sp_fit[current] = self.fit_func(self.sub_pop[current,:])
		self.num_evals += 1
		return 1


	def inner(self, ind1, ind2):
		sum = 0.0
		for i in range(self.dimen):
			sum += (ind1[i] * ind2[i] )
		return  sum

	def sort_population(self):

 
		dbest = 99
		for i in range(self.children + self.family):
			self.list[i] = i
		for i in range(self.children + self.family - 1):
			dbest = self.sp_fit[self.list[i]]
			for j in range(i + 1, self.children + self.family):
				if(self.sp_fit[self.list[j]]  < dbest):
					dbest = self.sp_fit[self.list[j]]
					temp = self.list[j]
					self.list[j] = self.list[i]
					self.list[i] = temp

	def replace_parents(self): #here the best (1 or 2) individuals replace the family of parents

 
		for j in range(self.family):
			self.population[ self.parents[j],:]  =  self.sub_pop[ self.list[j],:] # Update population with new species
			fx = self.fit_func(self.population[ self.parents[j],:])
			self.fitness[self.parents[j]]   =  fx
			self.num_evals += 1

	def family_members(self): #//here a random family (1 or 2) of parents is created who would be replaced by good individuals

 
		swp = 0
		for i in range(self.pop_size):
			self.parents[i] = i
		for i in range(self.family):
			randomIndex = random.randint(0, self.pop_size - 1) + i # Get random index in population
			if randomIndex > (self.pop_size-1):
				randomIndex = self.pop_size-1
			swp = self.parents[randomIndex]
			self.parents[randomIndex] = self.parents[i]
			self.parents[i] = swp

	def find_parents(self): #here the parents to be replaced are added to the temporary subpopulation to assess their goodness against the new solutions formed which will be the basis of whether they should be kept or not
		self.family_members()
 
		for j in range(self.family):
			self.sub_pop[self.children + j, :] = self.population[self.parents[j],:]
			fx = self.fit_func(self.sub_pop[self.children + j, :])
			self.sp_fit[self.children + j]  = fx
			self.num_evals += 1

	def random_parents(self ):

 


		for i in range(self.pop_size):
			self.temp_index[i] = i

		swp=self.temp_index[0]
		self.temp_index[0]=self.temp_index[self.best_index]
		self.temp_index[self.best_index]  = swp
		 #best is always included as a parent and is the index parent
		  # this can be changed for solving a generic problem
		for i in range(1, self.rand_parents):
			index= np.random.randint(self.pop_size)+i
			if index > (self.pop_size-1):
				index = self.pop_size-1
			swp=self.temp_index[index]
			self.temp_index[index]=self.temp_index[i]
			self.temp_index[i]=swp

	def evolve(self, outfile   ):
		#np.savetxt(outfile, self.population, fmt = '%1.2f' )

 

		#pop = np.loadtxt("pop.txt" )
		#genIndex = np.loadtxt("out3.txt" )
		#mom = np.loadtxt("out2.txt" )
		#self.population = pop
		tempfit = 0
		prevfitness = 99
		self.evaluate()
		tempfit= self.fitness[self.best_index]
		while(self.num_evals < self.max_evals):
			tempfit = self.best_fit
			self.random_parents()
			for i in range(self.children):
				tag = self.parent_centric_xover(i)
				if (tag == 0):
					break
			if tag == 0:
				break
			self.find_parents()
			self.sort_population()
			self.replace_parents()
			self.best_index = 0
			tempfit = self.fitness[0]
			for x in range(1, self.pop_size):
				if(self.fitness[x] < tempfit):
					self.best_index = x
					tempfit  =  self.fitness[x]
			if self.num_evals % 197 == 0:
				#print(self.population[self.best_index])            
				print(self.fitness[self.best_index], ' fitness')
				print(self.num_evals, 'num of evals\n\n\n')
			np.savetxt(outfile, [ self.num_evals, self.best_index, self.best_fit], fmt='%1.5f', newline="\n")
		print(self.sub_pop, '  sub_pop')
		#print(self.population[self.best_index], ' best sol'   )              
		print(self.fitness[self.best_index], ' fitness')


def main():


	problem = 3


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

	# print(topology, ' topology')

	netw = topology


	y_test =  testdata[:,netw[0]]
	y_train =  traindata[:,netw[0]]


	outfile=open('pop_.txt','w')
 
 
	random.seed(time.time())
	max_evals = 10000
	pop_size =  100
	num_varibles = (netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] + netw[2]  # num of weights and bias
	max_limits = np.repeat(50, num_varibles)
	print(max_limits, ' max_limits')
	min_limits = np.repeat(-50, num_varibles)


	g3pcx  = evolution(pop_size, num_varibles, max_evals,  max_limits, min_limits, netw, traindata, testdata)
	 
	g3pcx.evolve(outfile)

 














if __name__ == "__main__": main()
