#include "bits/stdc++.h"
#include <pthread.h>
using namespace std;

int thread_num;
long long int toss_num;
void* child(void* data);
int main(int argc, char *argv[])
{
		thread_num = atoi(argv[1]);
		toss_num = atoll(argv[2]);
		vector<pthread_t> t(thread_num);
		int data[thread_num];
		for(int i=0; i<thread_num; i++){
			data[i] = i;
		}

		for(int i=0; i < thread_num; i++){
			pthread_create(&t[i], NULL, child, (void*) &data[i]);
		}

		long long int res = 0;
		void* ret; 
		for(int i=0; i < thread_num; i++){
			pthread_join(t[i], &ret);
			res += *(long long int *) ret;
		}
		double pi_estimate = 4 * res /((double) toss_num);
		cout << pi_estimate << endl;
    return 0;
}

void* child(void* data){
	double prec = (double)1.0 / RAND_MAX;
	long long int* counter = (long long int*) malloc(sizeof(long long int));
	counter[0] = 0;
	long long int i = (long long int) *(int*) data;
	unsigned int seed = i;
	for(; i < toss_num; i += thread_num){	
		double x = rand_r(&seed) * prec;
		double y = rand_r(&seed) * prec;
		double distance_squared = x * x + y * y;
		if(distance_squared <= 1)
       counter[0]++;
	}
	return (void*) counter;
}
