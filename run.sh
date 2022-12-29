mpicc kmeans.c -o kmeans 
mpirun -np 4 ./kmeans $@