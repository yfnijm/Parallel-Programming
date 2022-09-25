#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include<bits/stdc++.h>
using namespace std;

MPI_Request requests[16];
MPI_Status status[16];
void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr,
                        int **a_mat_ptr, int **b_mat_ptr){
	std::ios::sync_with_stdio(false);
    std::cin.tie(0);	
	int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	if(world_rank == 0){
		//cin >> *n_ptr >> *m_ptr >> *l_ptr;
		scanf("%d %d %d", n_ptr, m_ptr, l_ptr);
	}
	MPI_Bcast(n_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(m_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(l_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int asize = (*n_ptr) * (*m_ptr), bsize = (*m_ptr) * (*l_ptr);
	*a_mat_ptr = (int *) malloc(sizeof(int) * asize);
	*b_mat_ptr = (int *) malloc(sizeof(int) * bsize);
	
	if(world_rank == 0){
		for(int i=0; i<asize; i++)
			scanf("%d", &(*a_mat_ptr)[i]);
	}
	MPI_Bcast(*a_mat_ptr, asize, MPI_INT, 0, MPI_COMM_WORLD);

	if(world_rank == 0){
		for(int i=0; i<bsize; i++)
			scanf("%d", &(*b_mat_ptr)[i]);
	}
	MPI_Bcast(*b_mat_ptr, bsize, MPI_INT, 0, MPI_COMM_WORLD);
}


void matrix_multiply(const int n, const int m, const int l,
                     const int *a_mat, const int *b_mat){
	int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	
	//int* res = (int*)calloc(n *l , sizeof(int));
	int* res = (int*) malloc(sizeof(int) * n * l);
	//int* res_tmp = (int*) malloc(sizeof(int) * n * l);
	for(int i=0; i<world_size; i++)
		requests[i] = MPI_REQUEST_NULL;


	int range = n * l / world_size;
	for(int i=world_rank * range; i < (world_rank + 1) * range; i++){
		res[i] = 0;
		int x = i / l, y = i % l;
		for(int j=0; j<m; j++){
			//if(i==0)cout << res[i];
			res[i] += a_mat[x * m + j] * b_mat[j * l + y];
		}
	}
	//cout<< world_rank*range << " " <<( world_rank+1 ) *range;
	if(world_rank > 0){
		MPI_Send(&res[world_rank * range], range, MPI_INT, 0, world_rank, MPI_COMM_WORLD);
	}else {
		for(int i = world_size * range; i < n * l; i++){
			res[i] = 0;
			int x = i / l, y = i % l;
			for(int j=0; j<m; j++){
				res[i] += a_mat[x * m + j] * b_mat[j * l + y];
			}
		}

		for(int i = 1; i < world_size; i++){
			MPI_Irecv(&res[i * range], range, MPI_INT, i, i, MPI_COMM_WORLD, &requests[i]);
		}
	}
	MPI_Waitall(world_size, requests, MPI_STATUSES_IGNORE);
	//MPI_Barrier(MPI_COMM_WORLD);	
	//MPI_Reduce(res, res_tmp, n*l, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	if(world_rank == 0){
		for(int i=0; i < n * l; i++){
			printf("%d ", res[i]);
			if(!((i+1) % l)) printf("\n");
		}
	}
	//MPI_Barrier(MPI_COMM_WORLD);	
}

void destruct_matrices(int *a_mat, int *b_mat){
	free(a_mat);
	free(b_mat);
}
