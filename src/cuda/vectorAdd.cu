/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <stdint.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>

#include "cuda_utils.h"

#define min(x,y) (x>y?x:y)
#define N 784

#define THREAD_PER_BLOCK 16

//smallest multiple of threadsPerBlock that is greater than or equal to N
#define BLOCK_PER_GRID min(32 , (N+THREAD_PER_BLOCK-1) / THREAD_PER_BLOCK )

/**
 * @brief Core unit of the neural network (neuron and synapses)
 */
struct Cuda_Cell{
	int n_inputs;
	double *input;
	double *weight;
	double output;
	double bias;
};

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
	__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		C[i] = A[i] + B[i];
	}
}

__global__ void printInput(const double *V1, const int size)
{
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x ;
	printf("Input: Hola desde kernel blockdim %d blockidx %d thidx %d element[%d] %f\n", blockDim.x, blockIdx.x, threadIdx.x, tid, V1[tid]);
}

__global__ void vectorDotProduct(const double *V1, const double *V2, double *V3)
{
	//Guarda la suma de cada thread
	__shared__ double chache[THREAD_PER_BLOCK] ;
	double temp = 0;
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x ;
	unsigned int chacheindex = threadIdx.x ;

	//printf("Hola desde kernel blockdim %d blockidx %d thidx %d\n", blockDim.x, blockIdx.x, threadIdx.x);
	while ( tid < N ) {
		temp += V1[tid] * V2[tid] ;
		//printf("(%d, %d, %d) tid %d .. %f %f temp %f\n", blockDim.x, blockIdx.x, threadIdx.x, tid, V1[tid], V2[tid], temp);
		tid += blockDim.x * gridDim.x;
	}

	chache[chacheindex] = temp;
	__syncthreads(); //Espero a que todo termine

	int i  = blockDim.x / 2 ;
	//printf("i %d Cache block %d th %d %f\n", i, blockIdx.x, threadIdx.x, chache[chacheindex]);
	while(i != 0) {
		if(chacheindex < i)
			chache[chacheindex] += chache[chacheindex + i];
		//printf("sum i %d Cache block %d th %d %f\n", i, blockIdx.x, threadIdx.x, chache[chacheindex]);

		__syncthreads();
		i /= 2 ;
	}

	if(chacheindex == 0) {
		V3[blockIdx.x] = chache[0];
		printf("V3[%d] %f\n", blockIdx.x, V3[blockIdx.x]);
	}
}

/**
 * @details Initialize layer by setting all weights to random values [0-1]
 * @attention It actually makes no difference whether the weights are
 * initialized to a constant (e.g. 0.5) or to a random number.
 * The result (85% success rate) will not change significantly.
 */
extern "C" int cuda_init_layer(Cuda_Layer *l, int n_input_cells, int n_output_cells)
{
	cudaError_t err = cudaSuccess;
	l->cell = (Cuda_Cell*)calloc(1, sizeof(Cuda_Cell) * n_output_cells);
	l->cell->n_inputs = n_input_cells;
	for(int i = 0; i < n_output_cells; i++) {
		err = cudaMalloc((void **)&l->cell[i].input, sizeof(double) * n_input_cells);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to allocate input vec (error code %s)!\n", cudaGetErrorString(err));
			return -1;
		}
		err = cudaMalloc((void **)&l->cell[i].weight, sizeof(double) * n_input_cells);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to allocate input weight (error code %s)!\n", cudaGetErrorString(err));
			//TODO Liberar y salir bien
			return -1;
		}
	}

	double *aux;
	aux = (double*)malloc(n_input_cells * sizeof(double));

	for (int o = 0; o < n_output_cells; o++){
		for (int i = 0; i < n_input_cells; i++){
			//TODO Inicializar en cuda
			//l->cell[o].input[i]=0;
			//l->cell[o].weight[i]=rand()/(double)(RAND_MAX);
			//aux[i] = rand()/(double)(RAND_MAX);
			aux[i] = 0.5;
		}
		err = cudaMemcpy(l->cell[o].weight, aux, n_input_cells * sizeof(double), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to copy input from host to device cell (error code %s)!\n", cudaGetErrorString(err));
			return -1;
		}
		l->cell[o].output = 0; //FIXME redundante
		l->cell[o].bias = 0; //FIXME redundante
	}
	//printInput<<<n_input_cells/16, 16>>>(l->cell[0].weight, n_input_cells);
	free(aux);

	return 0;
}

int cuda_set_cell_input(Cuda_Cell *c, MNIST_Image *img)
{
	cudaError_t err = cudaSuccess;
	double *aux;

	aux = (double*)malloc(c->n_inputs * sizeof(double));
	for(int i=0; i < c->n_inputs; i++){
		aux[i] = img->pixel[i] ? 1 : 0;
	}

	err = cudaMemcpy(c->input, aux, c->n_inputs * sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy input from host to device cell (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}
	free(aux);

	return 0;
}

void cuda_calc_cell_output(Cuda_Cell *c)
{
	double *V3_H, *V3_D;
	double sum = 0;

	c->output=0;

	printf("%d %d\n", THREAD_PER_BLOCK, BLOCK_PER_GRID);
	V3_H = (double *)calloc(1, sizeof(double) * BLOCK_PER_GRID);
	cudaMalloc((void **)&V3_D, BLOCK_PER_GRID*sizeof(double));

	cudaDeviceSynchronize();
	vectorDotProduct <<<BLOCK_PER_GRID, THREAD_PER_BLOCK>>> (c->input, c->weight, V3_D);
	cudaDeviceSynchronize();
	cudaMemcpy(V3_H, V3_D, BLOCK_PER_GRID*sizeof(double), cudaMemcpyDeviceToHost);

	for(int i = 0; i < BLOCK_PER_GRID; i++ )
		sum += V3_H[i];

	c->output = sum / c->n_inputs; // normalize output (0-1)
	fprintf(stderr, "%s:: output %f %f\n", __func__, sum, c->output);
}

extern "C" int cuda_train_cell(Cuda_Layer *l, int n_cell, MNIST_Image *img, int target)
{
	int ret;
	Cuda_Cell *c;
	c = &l->cell[n_cell];
	ret = cuda_set_cell_input(c, img);
	if(ret) {
		return -1;
	}
	cuda_calc_cell_output(c);
//
//	// learning (by updating the weights)
//	double err = getCellError(c, target);
//	updateCellWeights(c, err);
	return 0;
}

extern "C" int copy_to_cuda(uint8_t *buf, int size)
{
	uint8_t *dev_buf = NULL;
	cudaError_t err = cudaSuccess;
	err = cudaMalloc((void **)&dev_buf, size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}
	err = cudaFree(dev_buf);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}
	return 0;
}

/**
 * Host main routine
 */
int
old_main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);

    // Allocate the host input vector B
    float *h_B = (float *)malloc(size);

    // Allocate the host output vector C
    float *h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;
}

