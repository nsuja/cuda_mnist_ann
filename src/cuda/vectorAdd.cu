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

typedef struct Cuda_Network	 Cuda_Network;
typedef struct Cuda_Layer	 Cuda_Layer;
typedef struct Cuda_Node	 Cuda_Node;

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
 * @brief Dynamic data structure modeling a neuron with a variable number of connections/weights
 */
struct Cuda_Node {
	double bias;
	double output;
	int wcount;
	double weights[];
};

/**
 * @brief Dynamic data structure holding a definable number of nodes that form a layer
 */
struct Cuda_Layer {
	int ncount;
	Node nodes[];
};

/**
 * @brief Dynamic data structure holding the whole network
 */
struct Cuda_Network{
	int i_node_size;
	int i_layer_size;
	int h_node_size;
	int h_layer_size;
	int o_node_size;
	int o_layer_size;
	double learningRate;         ///< Factor by which connection weight changes are applied
	ActFctType hidLayerActType;
	ActFctType outLayerActType;
	Layer layers[];
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

	__global__ void
vectorUpdateWeight(const double *input, double *weight, double err)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if(tid < N) {
		//printf("old weight[%d] %f\n", tid, weight[tid]);
		weight[tid] += input[tid] * err * 0.05;
		//printf("new weight[%d] %f\n", tid, weight[tid]);
		tid += blockDim.x * gridDim.x;
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
		//printf("V3[%d] %f\n", blockIdx.x, V3[blockIdx.x]);
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
	l->n_output = n_output_cells;
	l->cell = (Cuda_Cell*)calloc(1, sizeof(Cuda_Cell) * n_output_cells);
	for(int i = 0; i < n_output_cells; i++) {
		l->cell[i].n_inputs = n_input_cells;
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
			//l->cell[o].input[i] = 0;
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
		//printf("[%d]%f ", i, aux[i]);
	}
	//printf("\n");

	err = cudaMemcpy(c->input, aux, c->n_inputs * sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy input from host to device cell (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}
	//printInput<<<BLOCK_PER_GRID, THREAD_PER_BLOCK>>>(c->input, c->n_inputs);
	free(aux);

	return 0;
}

void cuda_calc_cell_output(Cuda_Cell *c)
{
	double *V3_H, *V3_D;
	double sum = 0;

	c->output=0;

	//printf("%d %d\n", THREAD_PER_BLOCK, BLOCK_PER_GRID);
	V3_H = (double *)calloc(1, sizeof(double) * BLOCK_PER_GRID);
	cudaMalloc((void **)&V3_D, BLOCK_PER_GRID*sizeof(double));

	cudaDeviceSynchronize();
	vectorDotProduct <<<BLOCK_PER_GRID, THREAD_PER_BLOCK>>> (c->input, c->weight, V3_D);
	cudaDeviceSynchronize();
	cudaMemcpy(V3_H, V3_D, BLOCK_PER_GRID*sizeof(double), cudaMemcpyDeviceToHost);

	for(int i = 0; i < BLOCK_PER_GRID; i++ )
		sum += V3_H[i];

	c->output = sum / c->n_inputs; // normalize output (0-1)
	//fprintf(stderr, "%s:: output %f %f\n", __func__, sum, c->output);
}

/**
 * @details Returns the difference between a target value and the cell's ouput
 */
double get_cell_error(Cuda_Cell *c, int target)
{
	double err = target - c->output;
	return err;
}

/**
 * @details Updates a cell's weights based on given error and LEARNING_RATE
 */
void update_cell_weights(Cuda_Cell *c, double err)
{
	vectorUpdateWeight<<<BLOCK_PER_GRID, THREAD_PER_BLOCK>>> (c->input, c->weight, err);
}

extern "C" int cuda_train_cell(Cuda_Layer *l, int n_cell, MNIST_Image *img, int target)
{
	int ret;
	Cuda_Cell *c;
	c = &l->cell[n_cell];
	cudaDeviceSynchronize();
	ret = cuda_set_cell_input(c, img);
	if(ret) {
		return -1;
	}
	cuda_calc_cell_output(c);

	// learning (by updating the weights)
	double err = get_cell_error(c, target);
	update_cell_weights(c, err);
	cudaDeviceSynchronize();
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

extern "C" int cuda_get_layer_prediction(Cuda_Layer *l)
{
	double maxOut = 0;
	int maxInd = 0;

	for (int i=0; i < l->n_output; i++){
		if (l->cell[i].output > maxOut){
			maxOut = l->cell[i].output;
			maxInd = i;
		}
	}

	return maxInd;
}

/**
 * @brief Creates a dynamically-sized, 3-layer (INTPUT, HIDDEN, OUTPUT) neural network
 * @param in_count Number of nodes in the INPUT layer
 * @param hid_count Number of nodes in the HIDDEN layer
 * @param out_count Number of nodes in the OUTPUT layer
 */
Cuda_Network *cuda_create_network(int in_count, int hid_count, int out_count)
{
	//Size de input layer
	int i_node_size     = sizeof(Cuda_Node); //NO tiene weight porque se usa para la entrada
	int i_layer_size    = sizeof(Cuda_Layer) + (in_count * i_node_size);

	//Size de capa oculta
	int h_weight_count = in_count;
	int h_node_size     = sizeof(Cuda_Node) + (h_weight_count * sizeof(double));
	int h_layer_size    = sizeof(Cuda_Layer) + (hid_count * h_node_size);

	//Calculo tamanio para la salida
	int o_weigth_count = hid_count;
	int o_node_size     = sizeof(Cuda_Node) + (o_weigth_count * sizeof(double));
	int o_layer_size    = sizeof(Cuda_Layer) + (out_count * o_node_size);

	//Pido memoria para la red
	Cuda_Network *nn = (Cuda_Network*)malloc(sizeof(Cuda_Network) + i_layer_size + h_layer_size + o_layer_size);

	// Set/remember byte sizes of each component of the network
	nn->i_node_size     = i_node_size;
	nn->i_layer_size    = i_layer_size;
	nn->h_node_size     = h_node_size;
	nn->h_layer_size    = h_layer_size;
	nn->o_node_size     = o_node_size;
	nn->o_layer_size    = o_layer_size;

	// Initialize the network by creating the INPUT, HIDDEN and OUTPUT layer inside of it
	initNetwork(nn, in_count, hid_count, out_count);

	// Setting defaults
	setNetworkDefaults(nn);

	// Init connection weights with random values
	initWeights(nn, HIDDEN);
	initWeights(nn, OUTPUT);

	return nn;
}


