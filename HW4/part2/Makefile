all:
	#mpicxx -O3 main.cc matmul.cc -o matmul
	mpicxx -O3 -fno-asynchronous-unwind-tables -mavx2 -mavx -mfma -ffast-math -funsafe-math-optimizations -floop-parallelize-all -ftree-parallelize-loops=4 -fprefetch-loop-arrays -fno-tree-vectorize -fprofile-use -w main.cc matmul.cc -o matmul
clean:
	rm matmul
