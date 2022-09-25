#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
	//Platform layer
	///Query Platform(init)
	///Query Devices(init)

    cl_int status;
    int filterSize = filterWidth * filterWidth;
	int imageSize = imageHeight * imageWidth;

	///Command Queue
	cl_command_queue queue = clCreateCommandQueue(*context, *device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &status);
    CHECK(status, "Queue");

	//runtime layer
	///Create Buffers
	cl_mem inputBuffer = clCreateBuffer(*context, CL_MEM_USE_HOST_PTR, imageSize * sizeof(float), inputImage, &status);
    CHECK(status, "Mem1");
	cl_mem outputBuffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, imageSize * sizeof(float), NULL, &status);
    CHECK(status, "Mem2");
	cl_mem filterBuffer = clCreateBuffer(*context, CL_MEM_USE_HOST_PTR, filterSize * sizeof(int), filter, &status);
    CHECK(status, "Mem3");

	///Compile Program(init)
	///Compile Kernel
	cl_kernel kernel = clCreateKernel(*program, "convolution", &status);
    CHECK(status, "Kernel");

	///Set Arguments
    status |= clSetKernelArg(kernel, 0, sizeof(int), &filterWidth);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &filterBuffer);
    status |= clSetKernelArg(kernel, 2, sizeof(int), &imageHeight);
    status |= clSetKernelArg(kernel, 3, sizeof(int), &imageWidth);
    status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &inputBuffer);
    status |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &outputBuffer);
	//int halffilterSize = filterWidth / 2;
    //status |= clSetKernelArg(kernel, 6, sizeof(int), &halffilterSize);
	CHECK(status, "Set");
	
	///Execute Kernel
	size_t global_work_size[2] = {imageSize, 1};
	status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
	CHECK(status, "Exec");
	status = clFinish(queue);
	CHECK(status, "Sync");

	///Read back buffer
	status |= clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, imageSize * sizeof(float), outputImage, 0, NULL, NULL);
    CHECK(status, "Back");
	return ;
redo:
	hostFE(filterWidth, filter, imageHeight, imageWidth,
            inputImage, outputImage, device,
            context, program);
}
