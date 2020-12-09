# surrogate-neuroevolution
Distributed Bayesian Optimization - NeuroEvolution

## Code
We have three different sets of problems:
* First set includes Iris, Cancer and Chess problems : [pso_distributed](https://github.com/sydney-machine-learning/surrogate-neuroevolution/tree/master/pso_distributed).       
  * Run the file [pso_dist.py] using [run_pso.sh] for DSNE versions.                                                                                                          
  * Run the file [surr_revamp_syncswap.py] using [run_surr_revamp_syncswap.sh] for surrogate version- BONE.                                                                  
  * Run the file [surr_sch.py] using [run_surr_sch.sh] for surrogate version- BONE*.

* Second set features the MNIST problem using CNN : [pso_cnn](https://github.com/sydney-machine-learning/surrogate-neuroevolution/tree/master/pso_cnn).       
  * Run the file [pso_cnn.py] using [run_pso_cnn.sh] for DSNE versions.                                                                                                         
  * Run the file [surr_sampled_cnn.py] using [run_surr_sampled_cnn.sh] for surrogate version- BONE.                                                                  
  * Run the file [surr_cnn_sch.py] using [run_surr_cnn_sch.sh] for surrogate version- BONE*.
  
* Third set features the Time-Series problem : [pso_time_series](https://github.com/sydney-machine-learning/surrogate-neuroevolution/tree/master/pso_time_series).       
  * Run the file [pso_timeseries.py] using [run_pso_timeseries.sh] for DSNE versions.                                                                                          
  * Run the file [surr_pso_timeseries.py] using [run_surr_pso_timeseries.sh] for surrogate version- BONE.                                                                  
  * Run the file [surr_pso_ts_sch.py] using [run_surr_pso_ts_sch.sh] for surrogate version- BONE*.  
  
## Data
The Data used in Experiments can be found here: [DATA](https://github.com/sydney-machine-learning/surrogate-neuroevolution/tree/master/DATA)

## Prerequisites
Installation of libraries such as Keras, Tensorflow and scikitlearn is required for surrogate training, Pytorch is required for running the experiments for MNIST and Time-Series problems.

## Experiments
Sample results for all the problems can be found here: [results](https://github.com/sydney-machine-learning/surrogate-neuroevolution/tree/master/results). The files named "final.txt" report the final results including mean and standard deviation for different versions for different problems. 
