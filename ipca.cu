#include <stdio.h>
#include <jpeglib.h>
#include <cuda.h>
#include "ipca.h"

#define COUNT 300
#define DIMENSION 16
#define STRIDE 3072
#define AMNESIC 1.0
#define SIZE 32


/*
__global__ void kernel(int* tableFrame)
{
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        tableFrame[tid] = tid;
}
*/

__global__ void ipca_kernel( int current, int length, double* tableIn, double* tableU, double* tableV, int* tableFrame ) 
{
	double* imgA = (double*)malloc(sizeof(double)*STRIDE);
	double* imgB = (double*)malloc(sizeof(double)*STRIDE);
	double* imgC = (double*)malloc(sizeof(double)*STRIDE);

	///// thread id -> dimension id
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	tableFrame[tid] = -1;

	for(int f = -tid; f < length; f++)
	{
		__syncthreads();

		int frame = current + f;
		if(tid == 0) tableFrame[0] = frame;
		double* strideIn = tableIn + STRIDE * f;
		double* strideV = tableV + STRIDE * tid;
		double* strideU = tableU + STRIDE * tid;

		if(tid == 0)
		{
			for(int s=0; s<STRIDE; s++) strideU[s] = strideIn[s];
		}
		if(tid == frame)
		{
			for(int s=0; s<STRIDE; s++) strideV[s] = strideU[s];
			//tableFrame[tid] = f;
			continue;
		}
		if(tid > frame) continue;

		if(f == length/2) tableFrame[tid] = tableFrame[0];

		///// Vi(n) = [a= (n-1-l)/n * Vi(n-1)] + [b= (1+l)/n * Ui(n)T Vi(n-1)/|Vi(n-1)| * Ui(n) ]
		double nrmV = 0;
		for(int s=0; s<STRIDE; s++){ nrmV += strideV[s]*strideV[s]; }
		nrmV = sqrt(nrmV);
		double dotUV = 0;
		for(int s=0; s<STRIDE; s++){ dotUV += strideV[s]*strideU[s]; }
		double scalerA = ((double)frame - 1.0 - AMNESIC) / frame;
		double scalerB = (1.0 + AMNESIC) * dotUV / ((double)frame * nrmV);

		for(int s=0; s<STRIDE; s++) imgA[s] = strideV[s] * scalerA;
		for(int s=0; s<STRIDE; s++) imgB[s] = strideU[s] * scalerB;
		for(int s=0; s<STRIDE; s++) strideV[s] = imgA[s] + imgB[s];

		///// Ui+1(n) = Ui(n) - [c= Ui(n)T Vi(n)/|Vi(n)| * Vi(n)/|Vi(n)| ]
		if(tid >= DIMENSION - 1) continue;

		nrmV = 0;
		for(int s=0; s<STRIDE; s++){ nrmV += strideV[s]*strideV[s]; }
		nrmV = sqrt(nrmV);
		dotUV = 0;
		for(int s=0; s<STRIDE; s++){ dotUV += strideV[s]*strideU[s]; }
		double scalerC = dotUV / (nrmV * nrmV);

		for(int s=0; s<STRIDE; s++) imgC[s] = strideV[s] * scalerC;
		for(int s=0; s<STRIDE; s++) strideU[STRIDE + s] = strideU[s] - imgC[s];	
	}
	//tableFrame[tid] = length;
	free(imgA);
	free(imgB);
	free(imgC);
}

ipca_t* ipca_initialize()
{
	ipca_t* t = malloc(sizeof(ipca_t));
	t->COUNT = COUNT;
	t->STRIDE = STRIDE;
	t->DIMENSION = DIMENSION;
	t->frame = 0;

	t->hostImages = new double[STRIDE * COUNT];
	t->hostU = new double[STRIDE * DIMENSION];
	t->hostV = new double[STRIDE * DIMENSION];
	t->hostFrames = new int[DIMENSION];

	for(int p = 0; p < STRIDE * DIMENSION; p++)
	{
		t->hostU[p] = 0;
		t->hostV[p] = 0;
	}
	
	t->sizeImages = sizeof(double)*STRIDE*COUNT;
	t->sizeU = sizeof(double)*STRIDE*DIMENSION;
	t->sizeV = sizeof(double)*STRIDE*DIMENSION;
	t->sizeFrames = sizeof(int)*DIMENSION;
 	cudaMalloc(&t->deviceImages, t->sizeImages);
 	cudaMalloc(&t->deviceU, t->sizeU);
 	cudaMalloc(&t->deviceV, t->sizeV);
	cudaMalloc(&t->deviceFrames, t->sizeFrames);
        printf("0. %s\n", cudaGetErrorString(cudaGetLastError()));

	cudaMemcpy(t->deviceU, t->hostU, t->sizeU, cudaMemcpyHostToDevice);
	cudaMemcpy(t->deviceV, t->hostV, t->sizeV, cudaMemcpyHostToDevice);
        printf("1. %s\n", cudaGetErrorString(cudaGetLastError()));

	return t;
}

void ipca_run(ipca_t* t, double* images)
{
	clock_t start, end;
	start = clock();

	cudaMemcpy(t->deviceImages, images, t->sizeImages, cudaMemcpyHostToDevice);
	dim3 grid(1,1,1);
	dim3 block(16,1,1);
	ipca_kernel<<<grid, block>>>(t->frame, COUNT, t->deviceImages, t->deviceU, t->deviceV, t->deviceFrames);
	t->frame += COUNT;

	end = clock();
	printf("2. %fsec %s\n", (double)(end - start) /CLOCKS_PER_SEC, cudaGetErrorString(cudaGetLastError()));
}

void ipca_sync(ipca_t* t)
{
	cudaMemcpy(t->hostU, t->deviceU, t->sizeU, cudaMemcpyDeviceToHost);
	cudaMemcpy(t->hostV, t->deviceV, t->sizeV, cudaMemcpyDeviceToHost);
	cudaMemcpy(t->hostFrames, t->deviceFrames, t->sizeFrames, cudaMemcpyDeviceToHost);
        printf("3. %s\n", cudaGetErrorString(cudaGetLastError()));
}

void ipca_finalize(ipca_t* t)
{
        cudaFree(t->deviceImages);
        cudaFree(t->deviceU);
        cudaFree(t->deviceV);
        cudaFree(t->deviceFrames);

	delete(t->hostImages);
	delete(t->hostU);
	delete(t->hostV);
	delete(t->hostFrames);
	
	free(t);
}

