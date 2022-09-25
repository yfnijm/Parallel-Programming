#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#define GROUPSIZEX 5
#define GROUPSIZEY 3

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
	for(int i=0; i<GROUPSIZEX; i++)
		for(int j=GROUPSIZEY-1; j>= 0; j--){
			int thisX = (gridDim.x * i + blockIdx.x) * blockDim.x + threadIdx.x;// * i;//GROUPSIZEX + i;
			int thisY = (gridDim.y * j + blockIdx.y) * blockDim.y + threadIdx.y;// * j;//GROUPSIZEY + j;
			if(thisX >= resX || thisY >= resY) continue;
			float x = lowerX + thisX * stepX;
			float y = lowerY + thisY * stepY;
			device_memo[thisY * resX + thisX] = mandel(x, y, maxIterations);
		}
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
	//sort(img, img + resX * resY); 
	//20 / 1024
	//int blockSize, minGridSize;
	//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, mandelKernel, 0, resX * resY); 	
	//cout << width << minGridSize << " " << blockSize << endl;
	//
	//int blockSizeSqrt = (int) sqrt(blockSize);
	//int width = -blockSizeSqrt & blockSizeSqrt;
	int widthX = 16, widthY = 16;

	int gridX = (resX + (widthX * GROUPSIZEX) - 1) / (widthX * GROUPSIZEX);
	int gridY = (resY + (widthY * GROUPSIZEY) - 1) / (widthY * GROUPSIZEY);
	//cout << gridX << " " << gridY << endl;
	dim3 block(widthX, widthY);
	dim3 grid(gridX, gridY);

	int* host_memo = img; //(int*) malloc(resX * resY * sizeof(int));
	int* device_memo;
	cudaMalloc((void **)&device_memo, resX * resY * sizeof(int));

	mandelKernel <<< grid, block >>> (lowerX, lowerY, stepX, stepY, device_memo, resX, resY, maxIterations);
	cudaMemcpy(host_memo, device_memo, resX * resY * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(device_memo);

	//cout << resX << " " << resY << endl;
	//memcpy(img, host_memo, resX * resY * sizeof(int));
	//free(host_memo);
}
