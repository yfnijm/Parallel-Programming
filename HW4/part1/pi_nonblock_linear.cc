#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
	// --- DON'T TOUCH ---
	MPI_Init(&argc, &argv);
	double start_time = MPI_Wtime();
	double pi_result;
	long long int tosses = atoi(argv[1]);
	int world_rank, world_size;
	// ---

	// TODO: MPI init
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	long long int count = 0, len = tosses / world_size + ((tosses % world_size) > world_rank);
	unsigned int seed = world_rank;
	while(len --){
		double x = rand_r(&seed)/ (double) RAND_MAX;
		double y = rand_r(&seed)/ (double) RAND_MAX;
		if(x*x + y*y <= 1.0) count++;
	}

	if (world_rank > 0)
	{
		// TODO: MPI workers
		MPI_Send(&count, 1, MPI_LONG_LONG_INT, 0, world_rank, MPI_COMM_WORLD);
	}
	else if (world_rank == 0)
	{
		// TODO: non-blocking MPI communication.
		// Use MPI_Irecv, MPI_Wait or MPI_Waitall.
		MPI_Request requests[world_size - 1];
		MPI_Status status[world_size - 1];
		long long int tmp[world_size - 1];
		for(int i=1; i<world_size; i++){
			MPI_Irecv(&tmp[i - 1], 1, MPI_LONG_LONG_INT, i, i, MPI_COMM_WORLD, &requests[i - 1]);
		}
		MPI_Waitall(world_size - 1, requests, status);

		for(int i=1; i<world_size; i++)
			count += tmp[i - 1];
	}

	if (world_rank == 0)
	{
		// TODO: PI result
		pi_result = (double) count / tosses * 4;

		// --- DON'T TOUCH ---
		double end_time = MPI_Wtime();
		printf("%lf\n", pi_result);
		printf("MPI running time: %lf Seconds\n", end_time - start_time);
		// ---
	}

	MPI_Finalize();
	return 0;
}
