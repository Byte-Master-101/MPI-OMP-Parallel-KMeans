#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <omp.h>

#define CLUSTER_COUNT  	2
#define ITERATION_COUNT 10000

#define PRINT_ITERATIONS 0

// Contains point data
struct point {
  double x, y;
};

// Prints usage message then exits
void usage(const char *err)
{
	printf("%s\nUsage: ./kmeans 1,3 4,5 -2,4.43 42.3,2\n", err);

	exit(0);
}

// Performs KMeans
int kmeans(struct point *points, size_t point_count)
{
	// Arrays to contain cluster points
	double kx[CLUSTER_COUNT] = {0}; 	// X Pos for each cluster
	double ky[CLUSTER_COUNT] = {0};		// Y Pos for each cluster
	int kcount[CLUSTER_COUNT] = {0}; 	// Number of points contained in each cluster

	// Pick the cluster points according to the given points (at first)
	// NOTE: It would be better to make sure no points start at the same location
	for(int i = 0; i < CLUSTER_COUNT; i++) {
		kx[i] = points[i].x;
		ky[i] = points[i].y;
	}

	// Print the point info (rank only)
	printf("\n");
	for (size_t i = 0; i < point_count; i++)
		printf("Point %ld: %lf,%lf\n", i+1, points[i].x, points[i].y);

	// Start parallel region
	#pragma omp parallel shared(kx) shared(ky) shared(kcount) 
	{
		// For each iteration
		for (size_t i = 0; i < ITERATION_COUNT; i++)
		{
			// Print interations if requested
#if PRINT_ITERATIONS
			#pragma omp single 
			{
				printf("\n");
				for (size_t i = 0; i < CLUSTER_COUNT; i++)
					printf("%lf,%lf ", kx[i], ky[i]);
			}
#endif

			// Arrays to contain cluster points' partial values
			double l_kx[CLUSTER_COUNT] = {0}; 	// X Pos for each cluster
			double l_ky[CLUSTER_COUNT] = {0};	// Y Pos for each cluster
			int l_kcount[CLUSTER_COUNT] = {0}; 	// Number of points contained in each cluster

			#pragma omp for
			for (size_t j = 0; j < point_count; j++)
			{
				// Check which cluster this point belongs to
				size_t min_index = 0;
				double min_dist = DBL_MAX;
				for (size_t k = 0; k < CLUSTER_COUNT; k++)
				{
					double current_dist = ((kx[k] - points[j].x)*(kx[k] - points[j].x) + (ky[k] - points[j].y)*(ky[k] - points[j].y));
					if(current_dist < min_dist)
					{
						min_index = k;
						min_dist = current_dist;
					}
				}

				// Increase the cluster's sum and count (locally)
				l_kx[min_index] += points[j].x;
				l_ky[min_index] += points[j].y;
				l_kcount[min_index]++;
			}

			// Clear the cluster values after everybody is done
			#pragma omp barrier
			#pragma omp single 
			{
				for (size_t j = 0; j < CLUSTER_COUNT; j++)
				{
					kx[j] = 0;
					ky[j] = 0;
					kcount[j] = 0;
				}
			}

			// Add the new cluster values
			#pragma omp barrier
			#pragma omp critical
			{
				for (size_t j = 0; j < CLUSTER_COUNT; j++)
				{
					kx[j] += l_kx[j];
					ky[j] += l_ky[j];
					kcount[j] += l_kcount[j];
				}
			}

			// Calculate the new shared cluster value
			#pragma omp barrier
			#pragma omp single
			{
				// Calculate averages from the sums
				for (size_t j = 0; j < CLUSTER_COUNT; j++)
				{
					if(kcount[j] == 0) continue;

					kx[j] /= kcount[j];
					ky[j] /= kcount[j];
				}
			}
		}
	}

	// Print clusters
	printf("\n\n");
	for (size_t i = 0; i < CLUSTER_COUNT; i++)
		printf("Cluster %ld: %lf,%lf\n", i+1, kx[i], ky[i]);
}


int main(int argc, char** argv)
{
	// Check the existence of the args
	if(argc == 1)
		usage("\nPlease provide the points as arguments...\n");

	// Parse Args
	size_t point_count = argc-1;
	struct point *points = malloc(sizeof(struct point) * point_count);

	// Read points from args
	for (size_t i = 0; i < point_count; i++)
		if(sscanf(argv[1+i], "%lf,%lf", &points[i].x, &points[i].y) != 2)
			usage("\nAn argument is invalid...\n");
			
	// Perform KMeans
	kmeans(points, point_count);

	return 0;
}
