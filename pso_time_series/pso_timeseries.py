#........................................#
"""
PSO assisted neuroevolution of (Time-Series)Air-Pollution Dataset

"""
#........................................#
# !/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from scipy.linalg import sqrtm
from scipy.stats import skew,kurtosis
np.set_printoptions(suppress=True)
import random
from random import seed
import time
import math
import shutil
import multiprocessing 
import gc
import operator
import math
import copy    # array-copying convenience
import sys     # max float
import io
#pytorch
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import pickle
from sklearn import preprocessing
#torch.set_default_tensor_type('torch.DoubleTensor')
torch.backends.cudnn.enabled = False
device = 'cpu'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings 

# -----------------------------------
 
# using https://github.com/sydney-machine-learning/canonical_neuroevolution

#.....................................#

## Data loading ans pre-processing using torchvision for (Time-Series)Air-Pollution DataSet:

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

def data_load(X,y, train_size, batch_size,data='train'):
    if data == 'test':
        # transform to torch tensor
        tensor_x = torch.Tensor(np.expand_dims(X[train_size:, :, :], axis=1))
        tensor_y = torch.Tensor(y[train_size:, :])
        a = TensorDataset(tensor_x, tensor_y)

    else:
        # transform to torch tensor
        tensor_x = torch.Tensor(np.expand_dims(X[:train_size, :, :], axis=1))
        tensor_y = torch.Tensor(y[:train_size, :])
        a = TensorDataset(tensor_x, tensor_y)

    data_loader = torch.utils.data.DataLoader(
        a, batch_size=batch_size, shuffle=True)
    return data_loader

#.....................................#
#  nn.module defined using pytorch
class Model(nn.Module):
    def __init__(self, topo, lrate, batch_size, cnn_net='CNN'):
        super(Model, self).__init__()
        if cnn_net == 'CNN':
            self.conv1 = nn.Conv2d(1, 6, (2, 3))
            self.pool = nn.MaxPool2d(2, 2, padding=1)
            self.conv2 = nn.Conv2d(6, 16, 3)
            self.fc1 = nn.Linear(128, 50)
            self.fc2 = nn.Linear(50, 17)
            self.batch_size = batch_size
            #self.sigmoid = nn.Sigmoid()
            self.topo = topo
            self.los = 0
            self.criterion = torch.nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lrate)

    # Sequence of execution for the model layers
    
    def forward(self, x):
        #print("Stage1:",x.shape)
        x = F.relu(self.conv1(x))
        #print("Stage2:",x.shape)
        x = self.pool(x)
        #print("Stage3:",x.shape)
        x = F.relu(self.conv2(x))
        #print("Stage4:",x.shape)
        x = self.pool(x)
        #print("Stage5:",x.shape)
        x = torch.flatten(x, 1)
        #print("Stage6:",x.shape)
        x = F.relu(x)
        x = self.fc1(x)
        #print("Stage7:",x.shape)
        x = F.relu(x)
        x = self.fc2(x)        
        #print("Stage8:",x.shape)
        #interr
        return x
    # Used to apply softmax and obtain loss value

    def evaluate_proposal(self, data, w=None):
        self.los = 0
        if w is not None:
            self.loadparameters(w)
        flag = False
        y_pred = torch.zeros((len(data), self.batch_size, 17))
        for i, sample in enumerate(data, 0):
            inputs, labels = sample
            predicted = copy.deepcopy(self.forward(inputs).detach())
            # _, predicted = torch.max(a.data, 1)
            if(flag):
                y_pred = torch.cat((y_pred, predicted), dim=0)
            else:
              flag = True
              y_pred = predicted
            # y_pred[i] = predicted
            # b = copy.deepcopy(a)
            # prob[i] = self.softmax(b)
            loss = self.criterion(predicted, labels)
            self.los += loss
        self.los = torch.sqrt(self.los/len(data)) 
        #time.sleep(1.0)   
        return y_pred

    # Applied langevin gradient to obtain weight proposal

    def langevin_gradient(self, x, w=None):
        if w is not None:
            self.loadparameters(w)
        self.los = 0
        for i, sample in enumerate(x, 0):
            inputs, labels = sample
            outputs = self.forward(inputs)
            # _, predicted = torch.max(outputs.data, 1)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # if (i % 50 == 0):
            # print(loss.item(), ' is loss', i)
            self.los += copy.deepcopy(loss.item())
        self.los = np.sqrt(self.los/len(x))
        return copy.deepcopy(self.state_dict())

    # Obtain a list of the model parameters (weights and biases)

    def getparameters(self, w=None):
        l = np.array([1, 2])
        dic = {}
        if w is None:
            dic = self.state_dict()
        else:
            dic = copy.deepcopy(w)
        for name in sorted(dic.keys()):
            l = np.concatenate(
                (l, np.array(copy.deepcopy(dic[name])).reshape(-1)), axis=None)
        l = l[2:]
        return l

    # Loads the model parameters

    def loadparameters(self, param):
        self.load_state_dict(param)

    # Converting list of model parameters to pytorch dictionary form

    def dictfromlist(self, param):
        dic = {}
        i = 0
        for name in sorted(self.state_dict().keys()):
            dic[name] = torch.FloatTensor(param[i:i + (self.state_dict()[name]).view(-1).shape[0]]).view(
                self.state_dict()[name].shape)
            i += (self.state_dict()[name]).view(-1).shape[0]
        return dic

    # Adds random noise to weights to create new weight proposal

    def addnoiseandcopy(self, mea, std_dev):
        dic = {}
        w = self.state_dict()
        for name in (w.keys()):
            dic[name] = copy.deepcopy(
                w[name]) + torch.zeros(w[name].size()).normal_(mean=mea, std=std_dev)
        self.loadparameters(dic)
        return dic



class evaluate_neuralnetwork(object):  # class for fitness func 
    def __init__(self,  netw, traindata, testdata,batch_size,learn_rate,rnn):
        self.learn_rate = learn_rate # in case you wish to use gradients to help evolution
        self.batch_size = batch_size
        #self.neural_net = neuralnetwork(netw, traindata, testdata, learn_rate)  # FNN model, but can be extended to RNN model
        #self.rnn = Model(netw,learn_rate,batch_size,'CNN')
        self.rnn = rnn
        self.traindata = traindata
        self.testdata = testdata
        self.topology = netw
  

    # Updated
    def rmse(self, pred, actual):
        return self.rnn.los.item()
    
    # temporary accuracy: broken 
    """
    def accuracy(self,data,w):
        self.rnn.loadparameters(w)
        mse = 0
        for images, labels in data:
            labels = labels.to(device)
            predicted = self.rnn(images)
            print("Labels:", labels)
            print("Predicted:", predicted)
            interr
            mse += self.rnn.criterion(predicted, labels)
        print("mse:" , torch.sqrt(mse))
        return torch.sqrt(mse) 
    """
    
    # Updated
    def classification_perf(self, x, type_data):

        if type_data == 'train':
            data = self.traindata
        else:
            data = self.testdata
        
        flag = False
        for i, dat in enumerate(data, 0):
            inputs, labels = dat
            if(flag):
              y = torch.cat((y, labels), dim=0)
            else:
              y = labels
              flag = True
        fx = self.rnn.evaluate_proposal(data,x)
        fit = copy.deepcopy(self.rnn.los) 
        #acc = self.accuracy(data,x) 

        return fit, fit

    # Updated
    def fit_func(self, x , type_data):    #  function  (can be any other function, model or diff neural network models (FNN or RNN ))
        
        if type_data == 'train':
            data = self.traindata
        else:
            data = self.testdata
        flag = False
        for i, dat in enumerate(data, 0):
            inputs, labels = dat
            if(flag):
              y = torch.cat((y, labels), dim=0)
            else:
              y = labels
              flag = True
        
        fx = self.rnn.evaluate_proposal(data,x)
        fit = copy.deepcopy(self.rnn.los) 
        #return fit
        #acc = self.accuracy(data,x) 
        
        return fit # note we will minimize fitness, hence minimize error


    #Updated
    def neuro_gradient(self, data, w, depth = 1):  # BP with SGD (Stocastic BP)

        gradients = self.rnn.langevin_gradient(data,w)

        return self.rnn.getparameters(gradients)


class particle(evaluate_neuralnetwork):
    def __init__(self,  dim,  maxx, minx, netw, traindata, testdata, id_island, batch_size,learn_rate,rnn):
        self.rnn = rnn
        evaluate_neuralnetwork.__init__( self, netw, traindata, testdata,batch_size,learn_rate,self.rnn) # inherits neuroevolution class definition and methods
        
        #seed(id_island)
        #r_pos = np.asarray(random.sample(range(1, dim+1), dim) )/ (dim+1) #to force using random without np and convert to np (to avoid multiprocessing random seed issue)
        """
        np_pos = np.random.rand(dim)#/2 + r_pos/2
        np_vel = np.random.rand(dim)#/2 + r_pos/2
        self.position = (2 * np_pos -1) # using random.rand() rather than np.random.rand() to avoid multiprocesssing random issues
        self.velocity = (2* np_vel  -1)"""
        np_pos = self.rnn.addnoiseandcopy(0.0 , 0.005)
        np_vel = self.rnn.addnoiseandcopy(0.0 , 0.005)
        self.position = self.rnn.getparameters(np_pos)
        self.velocity = self.rnn.getparameters(np_vel)

        self.error =  self.fit_func(self.rnn.dictfromlist(self.position) , 'train') # curr error
        self.best_part_pos =  self.position.copy()
        self.best_part_err = self.error # best error 



class neuroevolution(evaluate_neuralnetwork, multiprocessing.Process):  # PSO http://www.scholarpedia.org/article/Particle_swarm_optimization
    def __init__(self,rnn, pop_size, lg_prob,dimen, max_evals,  max_limits, min_limits, netw, traindata, testdata, batch_size , learn_rate ,parameter_queue, wait_chain, event, island_id, swap_interval):
        
        multiprocessing.Process.__init__(self) # set up multiprocessing class
        self.rnn = rnn
        evaluate_neuralnetwork.__init__( self, netw, traindata, testdata,batch_size,learn_rate,self.rnn) # sepossiesiont up - inherits neuroevolution class definition and methods
        #self.rnn = Model(netw, learn_rate, batch_size, rnn_net= 'CNN')
        
        #multiprocessing Variables
        self.parameter_queue = parameter_queue
        self.signal_main = wait_chain
        self.event =  event
        self.island_id = island_id
        self.swap_interval = swap_interval
        # PSO Variables
        self.dim = dimen
        self.num_param = dimen
        self.n = pop_size
        self.minx = min_limits
        self.maxx = max_limits
        self.max_evals = max_evals
        self.minY = np.zeros((1,1))
        self.maxY = np.ones((1,1)) 
        self.EPSILON = 1e-40
        self.lg_prob = lg_prob
        # Network Variables
        self.netw = netw
        self.traindata = traindata
        self.testdata = testdata
        #Plotting variable
        self.plots = []
 
    def sort_swarm(self,swarm_list):
        for i in range(self.n -1):
            for j in range(0,self.n-1-i):
                if(swarm_list[j+1].best_part_err < swarm_list[j].best_part_err):
                    #swap the sample var
                    temp_var = swarm_list[j+1]
                    swarm_list[j+1] = swarm_list[j]
                    swarm_list[j] = temp_var 
        return swarm_list


    def run(self): # this is executed without even calling - due to multi-processing
        print("Dimension:" , self.num_param)
        netw = self.topology
        """
        w = self.rnn.state_dict()
        fx = self.rnn.evaluate_proposal(self.traindata , w)
        print("Rmse1:",self.rnn.los)
        w_new = self.rnn.langevin_gradient(self.traindata , w)
        print("Rmse2:", self.rnn.los)
        w_new2 = self.rnn.langevin_gradient(self.traindata , w_new)
        print("Rmse3:", self.rnn.los)
        """
        #PSO initialization starts
        np.random.seed(int(self.island_id) )
        swarm = [particle(self.dim, self.minx, self.maxx,  self.netw, self.traindata, self.testdata, self.island_id,self.batch_size,self.learn_rate,self.rnn) for i in range(self.n)] 
        
        best_swarm_pos = [0.0 for i in range(self.dim)] # not necess.
        #best_swarm_err = sys.float_info.max # swarm best
        best_swarm_err = swarm[0].error
        for i in range(self.n): # check each particle
            if swarm[i].error < best_swarm_err:
                best_swarm_err = swarm[i].error
                best_swarm_pos = copy.copy(swarm[i].position) 
        
        
        epoch = 0
        evals = 0
        w = 0.729    # inertia
        #c1 = 1.49445 # cognitive (particle)
        #c2 = 1.49445 # social (swarm)
        c1 = 1.4
        c2 = 1.4
        gradient_prob = self.lg_prob #0.1
        use_gradients = True
        
        #clear the event for the islands
        self.event.clear()
        count_real = 0
        print("Starting Generations")
        while evals < (self.max_evals ):
            #count_real = 0
            recalc = 0
            for i in range(self.n): # process each particle 
                #r_pos = np.asarray(random.sample(range(1, self.dim+1), self.dim) )/ (self.dim+1) #to force using random without np and convert to np (to avoid multiprocessing random seed issue)
                r1 = np.random.rand(self.dim)#/2 + r_pos/2
                r2 = np.random.rand(self.dim)

                swarm[i].velocity = ( (w * swarm[i].velocity) + (c1 * r1 * (swarm[i].best_part_pos - swarm[i].position)) +  (c2 * r2 * (best_swarm_pos - swarm[i].position)) )  
                
                swarm[i].position += swarm[i].velocity

                u = random.uniform(0, 1)
                depth = 1 # num of epochs for gradients by backprop
                #swarm[i].position += np.random.normal(0.0,0.005,self.num_param)
                if u < gradient_prob and use_gradients == True: 
                    swarm[i].position = self.neuro_gradient(self.traindata, self.rnn.dictfromlist(swarm[i].position.copy()), depth)  

                for k in range(self.dim): 
                    if swarm[i].position[k] < self.minx[k]:
                        swarm[i].position[k] = self.minx[k]
                    elif swarm[i].position[k] > self.maxx[k]:
                        swarm[i].position[k] = self.maxx[k]    
                    
                swarm[i].error = self.fit_func(self.rnn.dictfromlist(swarm[i].position),'train')

                if swarm[i].error < swarm[i].best_part_err:
                    swarm[i].best_part_err = swarm[i].error
                    swarm[i].best_part_pos = copy.copy(swarm[i].position)

                if swarm[i].error < best_swarm_err:
                    best_swarm_err = swarm[i].error
                    best_swarm_pos = copy.copy(swarm[i].position)
                           
            
            if evals % (self.n)  == 0: 

                train_per, rmse_train = self.classification_perf(self.rnn.dictfromlist(best_swarm_pos), 'train')
                test_per, rmse_test = self.classification_perf(self.rnn.dictfromlist(best_swarm_pos), 'test')
                print('evals_no:',evals,' ','epoch_no:', epoch,' ','island_id:',self.island_id,' ','train_perf:', float("{:.3f}".format(train_per)) ,' ','train_rmse:', float("{:.3f}".format(rmse_train)),' ' , 'classification_perf RMSE train * pso' ) 
                #if self.island_id == 1:
                #    self.plots.append(train_per)
                #print(evals, epoch, test_per ,  rmse_test, 'classification_perf  RMSE test * pso' )

            #time.sleep(0.5)    
            ## Sort according to fitness
            swarm = self.sort_swarm(swarm)
            exchange_param = [swarm[k].position for k in range((int)(self.n/5))]
                   
            if evals % self.swap_interval == 0 :
                #Parameter swapping starts
                #param = best_swarm_pos
                param = exchange_param
                self.parameter_queue.put(param)
                self.signal_main.set()
                self.event.clear()
                self.event.wait()
                result =  self.parameter_queue.get()
                #best_swarm_pos = result 
                #swarm[0].position = best_swarm_pos.copy()
                
                count_last = 0
                for k in range((int)(8*self.n/10),self.n):
                    swarm[k].position = result[count_last].copy()
                    count_last += 1
                

            epoch += 1
            evals += self.n
      
        train_per, rmse_train = self.classification_perf(self.rnn.dictfromlist(best_swarm_pos), 'train')
        test_per, rmse_test = self.classification_perf(self.rnn.dictfromlist(best_swarm_pos), 'test')
        #print(evals, epoch, train_per , rmse_train,  'classification_perf RMSE train * pso' )   
        #print(evals, epoch, test_per ,  rmse_test, 'classification_perf  RMSE test * pso' )
        file_name = 'island_results_2/island_'+ str(self.island_id)+ '.txt'
        np.savetxt(file_name, [train_per, rmse_train, test_per, rmse_test], fmt='%1.4f') 
        
 
        

        
class distributed_neuroevo:

    def __init__(self,  pop_size, max_evals, traindata, testdata, learn_rate, batch_size ,netw, num_islands,meth,lg_prob):
        #FNN Chain variables
        rnn = Model(netw, learn_rate, batch_size, cnn_net= 'CNN')
        self.rnn = rnn
        self.rnn_net = 'CNN'
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.traindata = traindata
        self.testdata = testdata
        self.topology = netw 
        self.pop_size = pop_size
        self.num_param =  len(rnn.getparameters(self.rnn.state_dict()))  # (topology[0] * topology[1]) + (topology[1] * topology[2]) + topology[1] + topology[2]
        self.max_evals = max_evals
        self.max_limits = np.repeat(10, self.num_param)
        self.min_limits = np.repeat(-10, self.num_param)
        self.meth = meth
        self.gradient_prob = lg_prob
        self.num_islands = num_islands
        self.islands = [] 
        self.island_numevals = int(self.max_evals/self.num_islands) 

        # create queues for transfer of parameters between process islands running in parallel 
        self.parameter_queue = [multiprocessing.Queue() for i in range(num_islands)]
        self.island_queue = multiprocessing.JoinableQueue()	
        self.wait_island = [multiprocessing.Event() for i in range (self.num_islands)]
        self.event = [multiprocessing.Event() for i in range (self.num_islands)]
        self.swap_interval = 2 * pop_size
        


    def initialize_islands(self):
        
        ## for the pso part
        if self.meth == 'PSO':
            for i in range(0, self.num_islands): 
                self.islands.append(neuroevolution(self.rnn,self.pop_size,self.gradient_prob ,self.num_param, self.island_numevals,  self.max_limits, self.min_limits, self.topology, self.traindata, self.testdata, self.batch_size, self.learn_rate ,self.parameter_queue[i],self.wait_island[i],self.event[i], i, self.swap_interval))
        """        
        ## for the DE part
        if self.meth == 'DE':
            for i in range(0, self.num_islands): 
                self.islands.append(neuroevolution_de(  self.pop_size,self.num_param, self.island_numevals,self.max_limits,self.min_limits, self.topology, self.traindata, self.testdata ,self.parameter_queue[i],self.wait_island[i],self.event[i], i, self.swap_interval))
        ## for the G3-PCX part
        if self.meth == 'G3PCX':
            for i in range(0, self.num_islands): 
                self.islands.append(neuroevolution_G3PCX(  self.pop_size,self.num_param, self.island_numevals,self.max_limits,self.min_limits, self.topology, self.traindata, self.testdata ,self.parameter_queue[i],self.wait_island[i],self.event[i], i, self.swap_interval))
        
        ## for the CMAES part
        if self.meth == 'CMAES':
            for i in range(0, self.num_islands): 
                self.islands.append(neuroevolution_cmaes(  self.pop_size,self.num_param, self.island_numevals,self.max_limits,self.min_limits, self.topology, self.traindata, self.testdata ,self.parameter_queue[i],self.wait_island[i],self.event[i], i, self.swap_interval))
        """
    #Function for swapping the islands upon reaching the swap interval for each chain.
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


        train_per, test_per, rmse_train, rmse_test, train_per_std, test_per_std, rmse_train_std, rmse_test_std = self.get_results()


        return   train_per, test_per, rmse_train, rmse_test, train_per_std, test_per_std, rmse_train_std, rmse_test_std






    def get_results(self):

        res_collect = np.zeros((self.num_islands,4))
        
         
        for i in range(self.num_islands):
            file_name = 'island_results_2/island_'+ str(i)+ '.txt'
            dat = np.loadtxt(file_name)
            res_collect[i,:] = dat 

        print(res_collect, ' res_collect')

        train_per = np.mean(res_collect[:,0])
        train_per_std = np.std(res_collect[:,0])

        rmse_train = np.mean(res_collect[:,1])
        rmse_train_std = np.std(res_collect[:,1])

        test_per = np.mean(res_collect[:,2])
        test_per_std = np.std(res_collect[:,2])

        rmse_test = np.mean(res_collect[:,3])
        rmse_test_std = np.std(res_collect[:,3])
            
        return   train_per, test_per, rmse_train, rmse_test, train_per_std, test_per_std, rmse_train_std, rmse_test_std



def main():


    problem = "Time-Series"
    print("Starting to run....")
    method = 'PSO'    # or 'G3PCX'#or 'DE' or 'CMAES'
    max_evals = int(sys.argv[1])
    lg_prob = float(sys.argv[2])
    if problem == "Time-Series":
        input_size = 320  
        hidden_size = 50  
        num_layers = 2  # Junk
        num_classes = 10
        batch_size = 16
        learn_rate = 0.1
        step_size = 10
        topology = [input_size, hidden_size, num_classes] 
        netw = topology
        name = 'CNN'
        load = False
        if(load):
            with open(('parameters/np_randomState.pickle'), "rb") as input_file:
                np.random.set_state(pickle.load(input_file))
            with open(('parameters/randomState.pickle'), "rb") as input_file:
                random.setstate(pickle.load(input_file))
        
        # Data taken from Sydney Machine Learning/ Bayesian CNN
        data = np.genfromtxt('DATA/time-series/ashok_mar19_mar20.csv', delimiter=',')[1:, 1:]
        data = preprocessing.normalize(data)
        X, y = split_sequences(data, step_size)
        X, y = shuffle_in_unison(X, y)
        train_size = 304
        #train_size = 20
        #Training data using the data loader
        traindata = data_load(X,y,train_size, batch_size,data='train')
        #print(traindata)
        
        #Testing data using the data loader
        testdata = data_load(X,y, train_size, batch_size,data='test')
        #print(testdata)

        examples = enumerate(traindata)
        batch_idx, (example_data, example_targets) = next(examples)
        print("Data.shape = ",example_data.shape)
        #path_results = 'results/' + problem +'/results_new.txt'
        #outfile_pso=open(path_results,'a+')
        path_temp = 'results/' + problem +'/temp.txt'
        outfile_temp = open(path_temp,'a+')
        #print(traindata)
        pop_size = 50
        num_islands = 10# currently testing on 10 islands using multiprocessing.
            
        timer = time.time()
        neuroevolution =  distributed_neuroevo(pop_size, max_evals, traindata, testdata, learn_rate, batch_size ,netw, num_islands,method,lg_prob)
        train_per, test_per, rmse_train, rmse_test, train_per_std, test_per_std, rmse_train_std, rmse_test_std = neuroevolution.evolve_islands()
        print('train_perf: ',float("{:.3f}".format(train_per)) ,'rmse_train: ' ,float("{:.3f}".format(rmse_train)),  'classification_perf RMSE train * pso' )   
        print('test_perf: ',float("{:.3f}".format(test_per)) , 'rmse_test: ' ,float("{:.3f}".format(rmse_test)), 'classification_perf  RMSE test * pso' )
        timer2 = time.time()
        timetotal = (timer2 - timer) /60
        allres =  np.asarray([train_per, test_per, timetotal,train_per_std, test_per_std]) 
        np.savetxt(outfile_temp,  allres   , fmt='%1.4f', newline='  '  )
        np.savetxt(outfile_temp,  [' PSO Gradient_Prob: '+str(lg_prob)], fmt="%s", newline=' \n '  )
            
          
     
if __name__ == "__main__": main()
