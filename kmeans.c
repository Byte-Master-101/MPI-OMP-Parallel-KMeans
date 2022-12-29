#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define CLUSTER_COUNT  	2
#define ITERATION_COUNT 10000

#define PRINT_ITERATIONS 0

int size, rank;

struct point {
  double x, y;
};

void usage(const char *err)
{
	if(rank == 0)
	{
		printf("%s\nUsage: ./kmeans 1,3 4,5 -2,4.43 42.3,2\n", err);
	}

	MPI_Finalize();
	
	exit(0);
}

int kmeans(struct point *points, size_t point_count)
{
	// Pick cluster points
	double l_kx[CLUSTER_COUNT], kx[CLUSTER_COUNT]; 			// X Pos for each cluster
	double l_ky[CLUSTER_COUNT], ky[CLUSTER_COUNT];			// Y Pos for each cluster

	double l_kcount[CLUSTER_COUNT], kcount[CLUSTER_COUNT]; 	// Number of points contained in each cluster

	// Pick the cluster points according to the given points (at first)
	// NOTE: It would be better to make sure no points start at the same location
	for(int i = 0; i < CLUSTER_COUNT; i++) {
		kx[i] = points[i].x;
		ky[i] = points[i].y;
	}

	// Print the point info (rank only)
	if(rank == 0)
	{
		printf("\n");

		for (size_t i = 0; i < point_count; i++)
		{
			printf("Point %ld: %lf,%lf\n", i+1, points[i].x, points[i].y);
		}
	}

	for (size_t i = 0; i < ITERATION_COUNT; i++)
	{
#if PRINT_ITERATIONS
		if(rank == 0)
		{
			printf("\n");
			for (size_t i = 0; i < CLUSTER_COUNT; i++)
				printf("%lf,%lf    ", i+1, kx[i], ky[i]);
		}
#endif
		
		// Send the current cluster positions
		MPI_Bcast(&kx, CLUSTER_COUNT, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&ky, CLUSTER_COUNT, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// Using interleaved assignment because it's easier to get your head around
		for (size_t j = rank; j < point_count; j += size)
		{
			// Check which cluster this point belongs to
			size_t min_index = 0;
			size_t min_dist = -1;
			for (size_t k = 0; k < CLUSTER_COUNT; k++)
			{
				size_t current_dist = (kx[k] - points[j].x)*(kx[k] - points[j].x) + (ky[k] - points[j].y)*(ky[k] - points[j].y);
				if(current_dist < min_dist)
				{
					min_index = k;
					min_dist = current_dist;
				}
			}

			l_kx[min_index] += points[j].x;
			l_ky[min_index] += points[j].y;
			l_kcount[min_index]++;
		}

		// Send the sums to root
		MPI_Reduce(&l_kx, &kx, CLUSTER_COUNT, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&l_ky, &ky, CLUSTER_COUNT, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&l_kcount, &kcount, CLUSTER_COUNT, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

		// Calculate averages from the sums
		for (size_t j = 0; j < CLUSTER_COUNT; j++)
		{
			kx[j] /= kcount[j];
			ky[j] /= kcount[j];
		}
	}

	if(rank == 0)
	{
		printf("\n\n");
		for (size_t i = 0; i < CLUSTER_COUNT; i++)
			printf("Cluster %ld: %lf,%lf\n", i+1, kx[i], ky[i]);
	}
}

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);

	// Initialisation
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(argc == 1)
		usage("\nPlease provide the points as arguments...\n");

	// Parse Args
	size_t point_count = argc-1;
	struct point *points = malloc(sizeof(struct point) * point_count);

	for (size_t i = 0; i < point_count; i++)
		if(sscanf(argv[1+i], "%lf,%lf", &points[i].x, &points[i].y) != 2)
			usage("\nAn argument is invalid...\n");
			
	kmeans(points, point_count);

	// Fin...
	MPI_Finalize();

	return 0;
}
