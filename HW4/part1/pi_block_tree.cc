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

    // TODO: binary tree redunction
   	int reducted_range = 1;
		while(reducted_range < world_size){
			if(world_rank % (reducted_range << 1)){
   			MPI_Send(&count, 1, MPI_LONG_LONG_INT, world_rank - reducted_range, world_rank, MPI_COMM_WORLD);
				break;
			}else {
				long long int tmp;
				MPI_Status status;
    		MPI_Recv(&tmp, 1, MPI_LONG_LONG_INT, 
						world_rank + reducted_range, world_rank + reducted_range, MPI_COMM_WORLD, &status);
				count += tmp;
			}
			reducted_range <<= 1;
		}

   	//MPI_Send(&count, 1, MPI_LONG_LONG_INT, 0, world_rank, MPI_COMM_WORLD);
    //MPI_Recv(&tmp, 1, MPI_LONG_LONG_INT, i, i, MPI_COMM_WORLD, &status);
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
