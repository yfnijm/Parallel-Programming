#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
using namespace std;
__device__ int mandel(float c_re, float c_im, int maxIteration)
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < maxIteration; ++i)
  {

    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int* device_memo, int resX, int resY, int maxIterations){
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;
	if(thisX >= resX || thisY >= resY) return ;
	
	float x = lowerX + thisX * stepX;
   	float y = lowerY + thisY * stepY;
	device_memo[thisY * resX + thisX] = mandel(x, y, maxIterations);
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
 
	//20 / 1024
	int blockSize, minGridSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, mandelKernel, 0, resX * resY); 	
	//cout << width << minGridSize << " " << blockSize << endl;
	//
	int blockSizeSqrt = (int) sqrt(blockSize);
	int width = -blockSizeSqrt & blockSizeSqrt;
	int gridX = (resX + width - 1) / width;
	int gridY = (resY + width - 1) / width;

	dim3 block(width, width);
	dim3 grid(gridX, gridY);

	int* host_memo = (int*) malloc(resX * resY * sizeof(int));
	int* device_memo;
	cudaMalloc((void **)&device_memo, resX * resY * sizeof(int));

	mandelKernel <<< grid, block >>> (lowerX, lowerY, stepX, stepY, device_memo, resX, resY, maxIterations);
	cudaMemcpy(host_memo, device_memo, resX * resY * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(device_memo);

	memcpy(img, host_memo, resX * resY * sizeof(int));
	free(host_memo);
}
