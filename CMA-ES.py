
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from scipy.linalg import sqrtm
from numpy.random import default_rng
np.set_printoptions(suppress=True)
import random
from random import seed
import time
import math
import os
import shutil

import multiprocessing 
import gc
  
import copy    # array-copying convenience
import sys     # max float
######################################
# CLASS for CMA-ES
class neuroevolution_cmaes(object): # https://en.wikipedia.org/wiki/CMA-ES
    def __init__(self, pop_size, dimen, max_evals,  max_limits, min_limits, netw, traindata, testdata, parameter_queue, wait_chain, event, island_id, swap_interval):
        
        multiprocessing.Process.__init__(self) # set up multiprocessing class

        evaluate_neuralnetwork.__init__( self, netw, traindata, testdata) 
        #multiprocessing variables
        self.parameter_queue = parameter_queue
        self.signal_main = wait_chain
        self.event =  event
        self.swap_interval = swap_interval
        self.island_id = island_id
        # architecture variables   
        self.dim = dimen
        self.pop_size = pop_size
        self.minx = min_limits
        self.maxx = max_limits
        self.max_evals = max_evals
        self.netw = netw
        self.traindata = traindata
        self.testdata = testdata
        ## cmaes variables
        #user defined parameters
        self.N = self.dim
        self.xmean = np.random.rand(self.N)
        self.sigma = 0.5
        #strategy parameter setting for selection
        #self.lmd = int(4 + np.floor(3 * np.log(self.N)))
        self.lmd = self.pop_size
        self.mu = np.floor((self.lmd)/2)
        ## particle variables
        self.var = np.random.rand(self.N , self.lmd)
        #self.fitness = np.random.rand(self.N , self.lmd)
        self.fitness = np.random.rand(self.lmd)
        # Initialize dynamic(internal) strategy parameters 
        self.pc = np.zeros(self.N) 
        self.ps = np.zeros(self.N)
        self.EPSILON = 1e-40
        
        self.B = np.eye(self.N , dtype = int)
        self.D = np.ones(self.N )
        self.eigeneval = 0
        
        self.C = np.matmul(np.matmul(self.B , np.diag(np.power(self.D , 2))), self.B.transpose())
        self.invsqrtC = np.matmul(np.matmul(self.B , np.diag(np.power(self.D , -1))), self.B.transpose())
        
        #self.C =  np.cov(self.var)
        #print("self.C:", self.C[0])
        #self.invsqrtC = np.sqrt(LA.inv(self.C))
        #rint("invsqrtC:", self.invsqrtC[0])
        #interrp
        self.chiN = np.power(self.N , 0.5) * (1-1/(4*self.N)+1/(21*self.N*self.N))
        

##sorting corrected
    def sort_samples(self):
        for i in range(self.lmd -1):
            for j in range(0,self.lmd-1-i):
                if(self.fitness[j+1] < self.fitness[j]):
                    #swap the sample var
                    temp_var = self.var[:,j]
                    self.var[:,j] = self.var[:,j+1]
                    self.var[:,j+1] = temp_var
                    #swap the corresponding fitness values
                    temp_fit = self.fitness[j]
                    self.fitness[j] = self.fitness[j+1]
                    self.fitness[j+1] = temp_fit



    def run(self):  # called automatically due to multiprocessing
        
        epoch = 0
        evals = 0
        countevals = 0
        weights = []
        sum_weights = 0.0
        sum_weights_norm_sq = 0.0
        #print("Value of mu is : " , self.mu)
        self.mu = int(self.mu)
        for i in range(self.mu):
            weights.append(1/self.mu)
            sum_weights += 1/self.mu
            #weights.append(np.log(self.mu + 0.5) - np.log(i+1))
            #sum_weights += np.log(self.mu + 0.5) - np.log(i+1)
        
        norm_weights = [x/sum_weights for x in weights]
        for i in range(self.mu):
            sum_weights_norm_sq += norm_weights[i] * norm_weights[i]
        mueff = 1/(sum_weights_norm_sq)
        #Strategy parameter setting for adaptation
        cc = (4+mueff/self.N) / (self.N+4 + 2*mueff/self.N)
        cs = (mueff+2) / (self.N+mueff+5)
        c1 = 2 / ((self.N+1.3) * (self.N+1.3)+mueff)
        cmu = min(1-c1, 2 * (mueff-2+1/mueff) / ((self.N+2)*(self.N+2)+mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(self.N+1))-1) + cs

        self.event.clear()

        while evals < (self.max_evals ):
            for k in range(self.lmd):
                rng = default_rng()
                #self.var[:,k] = self.xmean + self.sigma * np.matmul(self.B , np.multiply(self.D , np.random.randn(self.N)))
                self.var[:,k] = self.xmean + rng.multivariate_normal(np.zeros(self.N) ,self.sigma *self.sigma * self.C)
                #self.var[:,k] = np.nan_to_num(self.var[:,k])
                self.fitness[k] = self.fit_func(self.var[:,k])
                countevals += 1
                #evals += 1
            
            #print("Initial Fitness:",self.fitness,self.island_id)
            self.sort_samples()
            #print("Final fitness:",self.fitness,self.island_id)
            #interr
            #updation of the mean
            xold = self.xmean
            xmean =  np.zeros(self.N)
            for k in range(self.mu):
                xmean += norm_weights[i] * self.var[:,k]   
            #update evolution paths   
            
            self.ps = (1-cs)*self.ps + np.sqrt(cs*(2-cs)*mueff) * np.matmul(self.invsqrtC , (xmean-xold) / self.sigma)
            indicator = LA.norm(self.ps)/np.sqrt(1-np.power((1-cs),(2*countevals/self.lmd)))/self.chiN 
            if(indicator < 1.4 + 2/(self.N+1)):
                hsig = 1
            else:
                hsig = 0    
            
            self.pc = (1-cc)*self.pc + hsig * np.sqrt(cc*(2-cc)*mueff) * (xmean-xold) / self.sigma
            #update covariance matrix
            artmp = np.zeros((self.N,self.mu))
            for k in range(self.mu):
                artmp[:, k] = (self.var[:,k] - xold)/self.sigma
            norm_weights = np.array(norm_weights)
            self.C = (1-c1-cmu) * self.C + c1 * (self.pc* self.pc.T + (1-hsig) * cc*(2-cc) * self.C) + cmu * np.matmul(np.matmul(artmp , np.diag(norm_weights)) , artmp.T)
            #adapt step size 
            self.sigma = self.sigma * np.exp((cs/damps)*(LA.norm(self.ps)/self.chiN - 1))

            #self.invsqrtC = sqrtm(LA.inv(self.C))
            
            if ((countevals - self.eigeneval) > self.lmd/(c1+cmu)/self.N/10):
                self.eigeneval = countevals
                self.C = np.triu(self.C) + (np.triu(self.C,1)).T 
                self.D,self.B = LA.eig(self.C)    
                #self.D = np.sqrt(np.diag(self.D)) 
                self.D = np.sqrt(self.D)
                self.invsqrtC = np.matmul(np.matmul(self.B , np.diag(np.power(self.D , -1))), self.B.transpose())
                #self.invsqrtC = np.matmul(np.matmul(self.B , np.diag(np.power(self.D + self.EPSILON, -1))), self.B.transpose())
            


            best_param = copy.copy(self.var[:,0])

            time.sleep(0.1)  
    
            if evals % (self.pop_size )  == 0: 
                train_per, rmse_train = self.classification_perf(best_param, 'train')
                test_per, rmse_test = self.classification_perf(best_param, 'test')

                print('evals_no:',evals,' ','epoch_no:', epoch,' ','island_id:',self.island_id,' ','train_perf:', float("{:.3f}".format(train_per)) ,' ','train_rmse:', float("{:.3f}".format(rmse_train)),' ' , 'classification_perf RMSE train * cmaes' ) 
                                

            if (evals % self.swap_interval == 0 ):# interprocess (island) communication for exchange of neighbouring best
                param = best_param
                self.parameter_queue.put(param)
                self.signal_main.set()
                self.event.clear()
                print(1)
                self.event.wait()
                print(2)
                result =  self.parameter_queue.get()
                best_p = result 
                self.var[:,0] = best_p.copy()
                self.fitness[0] = self.fit_func(best_p.copy()) 
                

            epoch += 1
            evals += self.pop_size
        train_per, rmse_train = self.classification_perf(best_param, 'train')
        test_per, rmse_test = self.classification_perf(best_param, 'test')

        file_name = 'island_results_2/island_'+ str(self.island_id)+ '.txt'
        np.savetxt(file_name, [train_per, rmse_train, test_per, rmse_test], fmt='%1.4f') 
    
