# surrogate-neuroevolution
Surrogate-based distributed neuroevolution
* Initial Experiments:
For both the PSO and DE(Differential Evolution) parts only 2 islands have been initially considered in pso_de_dist_neuroevolutionfnn.py file :
* PSO :
For the pso part the code has been tested for different values of gradient proposals for considering Langevin Gradients as well as the swap proposals for swapping the islands:
num_islands = 2
Problem 3 = IRIS
Problem 5 = Cancer
* PSO:
 swap_proposal = 0, learn_rate = 0.1, gradient_probability = 0.1
 
 |problem |   run   | train_per | test_per | train_per_std | test_per_std | timetotal  |
 |--------|---------|-----------|----------|---------------|--------------|------------|
 |3.0000  | 1.0000  |   99.4445 |  95.7547 |    0.5555     |   2.3585     |  0.8843    | 
 |3.0000  | 2.0000  |   98.8889 |  96.2264 |    0.0000     |   0.0000     |  0.8531    |
 |3.0000  | 3.0000  |   98.3333 |  97.6191 |    0.5556     |   2.3809     |  0.8778    |
 
 swap_proposal =0 , learn_rate = 0.1 , gradient_probability = 0.5
 
 |problem |   run   | train_per | test_per | train_per_std  | test_per_std| timetotal |
 |--------|---------|-----------|----------|----------------|-------------|-----------|
 |3.0000  | 1.0000  |   99.4445 |   98.9474|    0.5555      |    1.0526   |  1.7638   |  
 |3.0000  | 2.0000  |   99.4445 |   96.3158|    0.5555      |    2.6316   |  1.8829   |  
 |3.0000  | 3.0000  |   98.8889 |   96.8421|    1.1111      |    3.1579   |  1.9359   | 
 
 swap_proposal = 0.5, learn_rate = 0.1, gradient_probability = 0.1
 
 |problem |   run   |  train_per|  test_per|  train_per_std | test_per_std|  timetotal| 
 |--------|---------|-----------|----------|----------------|-------------|-----------|
 |3.0000  | 1.0000  |   94.4445 |   88.7596|    5.5555      |    5.8139   |  0.9293   |  
 |3.0000  | 2.0000  |   100.0000|   96.5116|    0.0000      |    0.3876   |   0.8706  |    
 |3.0000  | 3.0000  |   98.3333 |   95.7364|    1.6666      |    1.9380   |   0.9557  |  
 
 swap_proposal = 0, learn_rate = 0.1, gradient_probability = 0.1
 
 |problem |   run   |  train_per|  test_per | train_per_std | test_per_std |  timetotal|
 |--------|---------|-----------|-----------|---------------|--------------|-----------|
 |5.0000  |  1.0000 |   97.6482 |  98.5714  |   0.3068      |  0.0000      |  4.5223   |  
 |5.0000  |  2.0000 |   97.6482 |  98.5714  |   0.3068      |  0.4762      |  4.7822   |  
 |5.0000  |  3.0000 |   97.6482 |  98.5714  |   0.3068      |  0.9524      |  4.6197   |  
 
 * DE(Differential Evolution):(crossp = 0.7 , mut = 0.8)
 problem =3 
 swap_proposal = 0.5
 
 |problem |   run   |  train_per | test_per | train_per_std | test_per_std | timetotal |
 |--------|---------|------------|----------|---------------|--------------|-----------|
 |3.0000  | 1.0000  |   93.8889  | 94.5652  |   1.6667      |    1.0869    | 0.6344    |  
 |3.0000  | 2.0000  |   96.1112  | 94.5652  |   0.5556      |    5.4348    | 0.6505    |  
 |3.0000  | 3.0000  |   94.4444  | 90.2174  |   1.1112      |    3.2609    | 0.7012    |  
 |3.0000  | 4.0000  |   93.3333  | 94.5652  |   2.2223      |    1.0869    | 0.7011    | 
 
 swap_proposal = 0
 
|problem |   run   | train_per  |test_per  |train_per_std  |test_per_std  |timetotal  |
|--------|---------|------------|----------|---------------|--------------|-----------|
|3.0000  | 1.0000  |   95.5556  | 92.2078  |   1.1112      |    0.0000    | 0.6475    |  
|3.0000  | 2.0000  |   95.5556  | 92.2078  |   1.1112      |    0.0000    | 0.6737    |  
|3.0000  | 3.0000  |   95.5556  | 92.2078  |   1.1112      |    0.0000    | 0.7395    |  
|3.0000  | 4.0000  |   95.5556  | 92.2078  |   1.1112      |    0.0000    | 0.7175    | 
 
 problem = 5
swap_proposal = 0

|problem |   run   | train_per  |test_per  |train_per_std  |test_per_std  |timetotal  |
|--------|---------|------------|----------|---------------|--------------|-----------|
| 5.0000 |  1.0000 |    96.2168 |  98.3333 |    0.1022     |     0.7143   |  3.1524   |   
| 5.0000 |  2.0000 |    96.2168 |  98.3333 |    0.1022     |     0.7143   |  3.3640   |   
| 5.0000 |  3.0000 |    96.2168 |  98.3333 |    0.1022     |     0.7143   |  3.3726   |   
| 5.0000 |  4.0000 |    96.2168 |  98.3333 |    0.1022     |     0.7143   |  3.3543   |   

swap_proposal = 0.5

|problem |   run   |  train_per | test_per | train_per_std | test_per_std | timetotal |
|--------|---------|------------|----------|---------------|--------------|-----------|
| 5.0000 |  1.0000 |    96.3190 |  97.1429 |    0.0000     |     0.4761   |  3.5182   |  
| 5.0000 |  2.0000 |    96.4213 |  97.3809 |    0.1022     |     0.2381   |  3.1787   |   
| 5.0000 |  3.0000 |    96.4213 |  98.3333 |    0.3067     |     0.2381   |  3.2082   |   
| 5.0000 |  4.0000 |    96.7280 |  97.6190 |    0.0000     |     0.0000   |  3.2191   |   
