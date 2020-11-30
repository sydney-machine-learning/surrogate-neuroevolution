#!/bin/sh 
echo Running all 	 
 
for max_evals in  10000 
 	do
	for lg_prob in  0.0 0.25
  		do
		for i in 1 2 3 4 5 
			do   
			python pso_cnn.py $max_evals $lg_prob
  
  
	done  
		
	done
done