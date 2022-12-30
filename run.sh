echo "========================= MPI ========================="
mpicc kmeans.c -o kmeans 
mpirun -np 4 ./kmeans $@

printf "\n\n========================= OPENMP ========================="
gcc -O0 -o kmeansomp -fopenmp kmeansomp.c &&
./kmeansomp $@