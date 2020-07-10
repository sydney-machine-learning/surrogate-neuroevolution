##########################################
# Class for G3PCX
class neuroevolution_G3PCX(evaluate_neuralnetwork, multiprocessing.Process):  #G3-PCX Evolutionary Alg by K Deb - 2002 (Real coded Genetic Alg)
    def __init__(self, pop_size, dimen, max_evals,  max_limits, min_limits, netw, traindata, testdata, parameter_queue, wait_chain, event, island_id, swap_interval):
        multiprocessing.Process.__init__(self) # set up multiprocessing class
        evaluate_neuralnetwork.__init__( self, netw, traindata, testdata)

        self.EPSILON = 1e-40  # convergence - not needed in this case
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
        ## the chain variables for multiprocessing
        self.swap_interval = swap_interval
        self.parameter_queue = parameter_queue
        self.signal_main = wait_chain
        self.event = event
        self.island_id = island_id
        self.plots = []
    

    def rand_normal(self, mean, stddev):
        if (not neuroevolution_G3PCX.n2_cached):
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
            neuroevolution_G3PCX.n2 = y*d
            # scale and translate to get desired mean and standard deviation
            result = n1*stddev + mean
            neuroevolution_G3PCX.n2_cached = True
            return result
        else:
            neuroevolution_G3PCX.n2_cached = False
            return neuroevolution_G3PCX.n2*stddev + mean

    def evaluate(self):

 

        self.fitness[0] = self.fit_func(self.population[0,:])  # this inherits method from neuroevolution class
        self.best_fit = self.fitness[0]
        for i in range(self.pop_size):
            self.fitness[i] = self.fit_func(self.population[i,:]) 
            if (self.best_fit> self.fitness[i]):
                self.best_fit =  self.fitness[i]
                self.best_index = i
        # self.num_evals += 1

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
        neuroevolution_G3PCX.n2 = 0.0
        neuroevolution_G3PCX.n2_cached = False
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
        # self.num_evals += 1
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
            # self.num_evals += 1

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
            # self.num_evals += 1

    def random_parents(self ):
        ## this function is basically for selecting the random parents from population 
        ## and always including the best 
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

    def run(self): # called due to multiprocessing without even calling 
        # Initial fit with best index.
        tempfit = 0
        # Number of epochs 
        epoch = 0
        #prevfitness = 99
        self.evaluate()
        tempfit= self.fitness[self.best_index]
        # starting the evaluations
        #tag = 1
        self.event.clear()
        while(self.num_evals < self.max_evals):
            tempfit = self.best_fit
            self.random_parents()
            #if tag == 1:
            for i in range(self.children):
                tag = self.parent_centric_xover(i)
                if (tag == 0):
                    break

            #if tag ==0:
                #time.sleep(0.5)        
            
            if tag == 1:
                self.find_parents()
                self.sort_population()
                self.replace_parents()
                self.best_index = 0
                tempfit = self.fitness[0]
                for x in range(1, self.pop_size):
                    if(self.fitness[x] < tempfit):
                        self.best_index = x
                        tempfit  =  self.fitness[x]

            #time.sleep(0.5)

            if (self.num_evals % (self.pop_size * 2) == 0): 
                train_per, rmse_train = self.classification_perf(self.population[self.best_index], 'train')
                test_per, rmse_test = self.classification_perf(self.population[self.best_index], 'test')
                print('tag:',tag,' ','evals_no:',self.num_evals,' ','epoch_no:',epoch,' ','island_id:',self.island_id,' ','train_perf:', float("{:.3f}".format(train_per)) ,' ','train_rmse:', float("{:.3f}".format(rmse_train)),' ' , 'perf RMSE train * g3pcx' )  
                #if self.island_id == 2:
                #    self.plots.append(train_per) 
                #print('\n')
                #print(self.fitness[self.best_index], ' fitness')
                #print(self.num_evals, 'num of evals\n\n\n') 

            # preparing for swapping     
            if (self.num_evals % self.swap_interval == 0 ): # interprocess (island) communication for exchange of neighbouring best_swarm_pos
                param = self.population[self.best_index]
                self.parameter_queue.put(param)
                self.signal_main.set()
                self.event.clear()
                print(1)
                self.event.wait()
                print(2)
                result =  self.parameter_queue.get()
                best_param  = result
                self.population[0] = best_param.copy()
                self.best_index = 0
                self.fitness[0] = self.fit_func(best_param.copy())
                self.best_fit = self.fitness[0] 

            self.num_evals += self.pop_size
            epoch += 1       


        train_per, rmse_train = self.classification_perf(self.population[self.best_index], 'train')
        test_per, rmse_test = self.classification_perf(self.population[self.best_index], 'test')

        file_name = 'island_results_2/island_'+ str(self.island_id)+ '.txt'
        np.savetxt(file_name, [train_per, rmse_train, test_per, rmse_test], fmt='%1.4f')
        #print(self.plots)
        #return train_per, test_per, rmse_train, rmse_test 
##########################################

