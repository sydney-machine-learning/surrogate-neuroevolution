#!/bin/sh 
echo Running all 	 
 
for problem in  3 4 5 6 8
 	do
	for lg_prob in  0.0 0.25 
  		do
		for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 #21 22 23 24 25 26 27 28 29 30
			do   
			python pso_dist.py $problem $lg_prob
  
  
	done  
		
	done
done
