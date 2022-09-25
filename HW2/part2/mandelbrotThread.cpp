#include <stdio.h>
#include <thread>
#include <mutex>
#include <vector>

#include "CycleTimer.h"

typedef struct
{
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int *output;
    int threadId;
    int numThreads;

		int startRow;
		int numRows;
} WorkerArgs;
std::vector<int> done;
std::mutex lock;

extern void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int numRows,
    int maxIterations,
    int output[]);

//
// workerThreadStart --
//
// Thread entrypoint.
void workerThreadStart(WorkerArgs *const args)
{

    // TODO FOR PP STUDENTS: Implement the body of the worker
    // thread here. Each thread should make a call to mandelbrotSerial()
    // to compute a part of the output image.  For example, in a
    // program that uses two threads, thread 0 could compute the top
    // half of the image and thread 1 could compute the bottom half.
		//mandelbrotSerial(args->x0, args->y0, args->x1, args->y1, args->width, args->height, 
				//args->startRow, args->numRows, args->maxIterations, args->output);
		for(int i=0; i<args->height; i++){
			int flag = 0;
			lock.lock();
				flag = done[i];
				done[i] = 1;
			lock.unlock();
			if(!flag) mandelbrotSerial(args->x0, args->y0, args->x1, args->y1, args->width, args->height,
				i, 1, args->maxIterations, args->output);
		}



		//mandelbrotSerial(args->x0, args->y0, args->x1, args->y1, args->width, args->height,
		//		args->startRow + args->numRows / 2, args->numRows - args->numRows / 2, args->maxIterations, args->output);

		//printf("Hello world from thread %d %ld\n", args->threadId, clock() - a);
}

//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by spawning std::threads.
void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    static constexpr int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    std::thread workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

		int dRow = height / numThreads;
		int remind = height % numThreads;
    for (int i = 0; i < numThreads; i++)
    {
        // TODO FOR PP STUDENTS: You may or may not wish to modify
        // the per-thread arguments here.  The code below copies the
        // same arguments for each thread
        
				args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;

				args[i].startRow = i ? args[i-1].startRow + args[i-1].numRows : 0;	
				args[i].numRows = dRow;
				if(remind){
					args[i].numRows++;
					remind--;
				}

        args[i].maxIterations = maxIterations;
        args[i].numThreads = numThreads;
        args[i].output = output;

        args[i].threadId = i;
    }
		done = std::vector<int>(height);
    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i = 1; i < numThreads; i++)
    {
        workers[i] = std::thread(workerThreadStart, &args[i]);
		}

    workerThreadStart(&args[0]);

    // join worker threads
    for (int i = 1; i < numThreads; i++)
    {
        workers[i].join();
    }
}
