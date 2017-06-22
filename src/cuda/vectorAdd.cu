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

#include <errno.h>
#include <stdio.h>
#include <stdint.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>

#include "cuda_utils.h"
#include <sys/time.h>
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>

#define MIN(x,y) (x < y ? x:y)

#define THREAD_PER_BLOCK (1024)
#define MAX_BLOCKS (10)

#define CUDA_LAYER_CANT (3)

#define CUDA_LEARNING_RATE_SIGMOID (0.2) //91.5%
__constant__ double _cuda_learning_rate;

#define NO_DEBUG_ALL (1)

#define cuda_free(x, ret) \
	if(cudaFree(x) != cudaSuccess) { \
		fprintf(stderr, "%s:: Fallo al liberar #x", __func__); \
		return ret; \
	}

//XXX Para que funcione el atomic add
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* a, double b) { return b; }
#endif

typedef struct Cuda_Layer	 Cuda_Layer;
typedef struct Cuda_Node	 Cuda_Node;

typedef enum {
	CUDA_ACT_SIGMOID,
	CUDA_ACT_TANH
} Cuda_Act_Func_Type;

typedef enum {
	CUDA_LAYER_INPUT,
	CUDA_LAYER_HIDDEN,
	CUDA_LAYER_OUTPUT
} Cuda_Layer_Type;

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
	int wcount;
	double *weights;
};

/**
 * @brief Dynamic data structure holding a definable number of nodes that form a layer
 */
struct Cuda_Layer {
	int n_output;
	double *targets[10];
	double *err_signal;
	double *err_sum;
	double *outputs;
	Cuda_Node *nodes;
};

/**
 * @brief Dynamic data structure holding the whole network
 */
struct Cuda_Network{
	double learning_rate;         ///< Factor by which connection weight changes are applied
	Cuda_Act_Func_Type hid_layer_act_type;
	Cuda_Act_Func_Type out_layer_act_type;
	double *super_input;
	Cuda_Layer **layers;
	cudaStream_t *streams;
};


int cuda_init_super_input(Cuda_Network *nn);
Cuda_Layer *cuda_get_layer(Cuda_Network *nn, Cuda_Layer_Type ltype);

uint64_t cu_get_time_usec()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000000ULL + tv.tv_usec;
}

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
template <unsigned int block_size>
__global__ void cu_dot(double * a_d, double * b_d, double * block_results_d,
		size_t size) {
	extern __shared__ int cache[];

	unsigned int tid = threadIdx.x;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	cache[tid] = 0;

	if(idx < size) {
		cache[tid] = a_d[idx] * b_d[idx];
	}

	__syncthreads();
	if(block_size >= 512) {
		if(tid < 256) {
			cache[tid] += cache[tid + 256];
		}
		__syncthreads();
	}

	if(block_size >= 256) {
		if(tid < 128) {
			cache[tid] += cache[tid + 128];
		}
		__syncthreads();
	}

	if(block_size >= 128) {
		if(tid < 64) {
			cache[tid] += cache[tid + 64];
		}
		__syncthreads();
	}

	if(tid < 32) {
		if(block_size >= 64) {
			cache[tid] += cache[tid + 32];
		}
		__syncthreads();

		if(block_size >= 32) {
			cache[tid] += cache[tid + 16];
		}
		__syncthreads();

		if(block_size >= 16) {
			cache[tid] += cache[tid + 8];
		}
		__syncthreads();

		if(block_size >= 8) {
			cache[tid] += cache[tid + 4];
		}
		__syncthreads();

		if(block_size >= 4) {
			cache[tid] += cache[tid + 2];
		}
		__syncthreads();

		if(block_size >= 2) {
			cache[tid] += cache[tid + 1];
		}
	}

	__syncthreads();
	if(tid == 0) {
		block_results_d[blockIdx.x] = cache[0];
	}
}



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
vectorUpdateWeight(const double *input, double *weight, const int size, double err)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if(tid < size) {
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

__global__ void vectorDotProduct(const double *V1, const double *V2, double *V3, const int size, int log)
{
	//Guarda la suma de cada thread
	__shared__ double chache[THREAD_PER_BLOCK] ;
	double temp = 0;
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x ;
	unsigned int chacheindex = threadIdx.x ;
	int stride = gridDim.x * blockDim.x;

	//printf("Hola desde kernel blockdim %d blockidx %d thidx %d\n", blockDim.x, blockIdx.x, threadIdx.x);
	while ( tid < size ) {
		temp += V1[tid] * V2[tid] ;
		//if(log)
		//	printf("(%d, %d, %d) weights %p dst %p tid %d .. %f %f temp %f\n", blockDim.x, blockIdx.x, threadIdx.x, V2, V3, tid, V1[tid], V2[tid], temp);
		tid += stride;
	}

	chache[chacheindex] = temp;
	__syncthreads(); //Espero a que todo termine

	int i  = blockDim.x / 2 ;
//	if(log)
//		printf("i %d Cache block %d th %d %f\n", i, blockIdx.x, threadIdx.x, chache[chacheindex]);
	while(i != 0) {
		if(chacheindex < i)
			chache[chacheindex] += chache[chacheindex + i];
		//if(log)
			//printf("sum i %d Cache block %d th %d %f\n", i, blockIdx.x, threadIdx.x, chache[chacheindex]);

		__syncthreads();
		i /= 2 ;
	}

	if(chacheindex == 0) {
		V3[blockIdx.x] = chache[0];
	}
}

__global__ void vectorGetErrorSum(const double *err_signal, const double *weights, double *error_sum, const int size, const int weight_n, const int err_n, int log)
{
	//Guarda la suma de cada thread
	__shared__ double cache[THREAD_PER_BLOCK] ;
	double temp = 0;
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x ;
	int stride = gridDim.x * blockDim.x;

	//printf("Hola desde kernel blockdim %d blockidx %d thidx %d\n", blockDim.x, blockIdx.x, threadIdx.x);
	for (int i = tid; i < size; i += stride) { //+1 para no activar el output de BIAS
		if(i % weight_n == 0)
			continue;
		if(log)
			printf("%s:: (%d, %d, %d) tid %d i %d(%d) err_n %d.. %f %f %f temp %f\n", __func__, blockDim.x, blockIdx.x, threadIdx.x, tid, i, i%weight_n, err_n, err_signal[(i/weight_n) + 1], weights[i], err_signal[(i/weight_n) + 1]* weights[i], error_sum[(i%weight_n) + 1]);

		temp += err_signal[(i/weight_n) + 1] * weights[i];
		//error_sum[(i%weight_n) + 1] += err_signal[(i/weight_n) + 1] * weights[i];
	}

	cache[threadIdx.x] = temp; //Guardo los resultados de todos los threads en una shared
	__syncthreads(); //Espero a que todo termine

	if(log && tid < size)
		printf("%s:: ..(%d, %d, %d) thidx %d tid %d (%d) err_n %d.. %f %f %f temp %f\n", __func__, blockDim.x, blockIdx.x, threadIdx.x, threadIdx.x, tid, tid%weight_n, err_n, err_signal[(tid/weight_n) + 1], weights[tid], err_signal[(tid/weight_n) + 1]* weights[tid], cache[threadIdx.x]);

	int i  = size / 2 ;
	int offset = 0;
	if(log && tid < size)
		printf("tid %d Cache block %d th %d %f\n", threadIdx.x, blockIdx.x, threadIdx.x, cache[threadIdx.x]);
	while(i >= weight_n) {
		if(i%weight_n == 0) {
			offset = 0;
		} else {
			offset = weight_n - i%weight_n;
		}
		if(threadIdx.x < (i-i%weight_n))
			cache[threadIdx.x] += cache[threadIdx.x+i+offset];

		__syncthreads();
		if(log && threadIdx.x < (i-i%weight_n))
			printf("i %d Cache block %d off %d .. %d %d %f\n", i, blockIdx.x, offset, threadIdx.x, threadIdx.x+i+offset, cache[threadIdx.x]);

		i = (i+offset)/2;
	}

	if(threadIdx.x < weight_n)
		error_sum[threadIdx.x] = cache[threadIdx.x];
}

//Horrible
__global__ void naiveGetMax(const double *vec, const int size, double *val, int *ind, int log)
{
	//Guarda la suma de cada thread
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x ;
	double max = -1;
	int pos = -1;

	if(tid == 0) {
		for(int i = 1; i < size; i++) {
			if(max < vec[i]) {
				max = vec[i];
				pos = i;
			}
		}
		__syncthreads();
		if(val)
			*val = max;
		if(ind)
			*ind = pos-1;
	}
}

__device__ __forceinline__ double sigmoid (double a)
{
	return 1.0 / (1.0 + exp (-a));
}

__global__ void sigmoid_kernel (const double * __restrict__ src, double * __restrict__ dst, int len)
{
	int stride = gridDim.x * blockDim.x;
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	for (int i = tid + 1; i < len; i += stride) { //+1 para no activar el output de BIAS
		dst[i] = sigmoid(src[i]);
	}
}

__global__ void vectorGetErrSignal(double *target, double *cur_output, double *err_output, const int cur_out_count, int is_delta, int log)
{
	//Guarda la suma de cada thread
	double delta = 0, deriv_val = 0;
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x ;
	int stride = gridDim.x * blockDim.x;

	//printf("Hola desde kernel blockdim %d blockidx %d thidx %d\n", blockDim.x, blockIdx.x, threadIdx.x);
	for (int i = tid + 1; i < cur_out_count; i += stride) { //+1 para no activar el output de BIAS
		if(!is_delta)
			delta = target[i] - cur_output[i];
		else
			delta = target[i];
		deriv_val = cur_output[i] * (1 - cur_output[i]);
		err_output[i] = delta * deriv_val;
		if(log)
			printf("Kernel i %d tid %d, target %f cur %f .. delta %f deriv %f ... err %f\n", i, tid, target[i], cur_output[i], delta, deriv_val, err_output[i]);
	}
}

__global__ void vectorUpdateWeights(double *weights, double *prev_outputs, double *err_signal, const int weight_count, int log)
{
	//Guarda la suma de cada thread
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x ;
	int stride = gridDim.x * blockDim.x;

	//printf("Hola desde kernel blockdim %d blockidx %d thidx %d\n", blockDim.x, blockIdx.x, threadIdx.x);
	for (int i = tid; i < weight_count; i += stride)
		//weights[i] += (CUDA_LEARNING_RATE_SIGMOID * prev_outputs[i] * (*err_signal));
		weights[i] += (_cuda_learning_rate * prev_outputs[i] * (*err_signal));

//	//BIAS
//	if(tid == 0) {
//		weights[tid] = weights[tid] + (CUDA_LEARNING_RATE_SIGMOID * 1.0 * (*err_signal));
//	} else {
//		//Pesos
//		for (int i = tid; i < weight_count; i += stride) { //+1 para no activar el output de BIAS
//			//if(log)
//			//	printf("Kernel i %d tid %d, RATE %f prev %f err %f .. off %f ... weight %f\n", i, tid, CUDA_LEARNING_RATE_SIGMOID, prev_outputs[i], *err_signal, CUDA_LEARNING_RATE_SIGMOID * prev_outputs[i] * (*err_signal), weights[i]);
//			//weights[i] += (_cuda_learning_rate * prev_outputs[i] * (*err_signal));
//			weights[i] += (CUDA_LEARNING_RATE_SIGMOID * prev_outputs[i] * (*err_signal));
//		}
//	}
}

void cuda_print_vector(FILE * fp, char *name, double *vec, int n)
{
	double *aux;
	if(NO_DEBUG_ALL)
		return;
	aux = (double *)malloc(sizeof(double) * n);
	cudaMemcpy(aux, vec, sizeof(double)*n, cudaMemcpyDeviceToHost);
	fprintf(fp, "%s:\n", name);
	for(int i = 0; i < n; i++) {
		fprintf(fp, "%1.06f ", i, aux[i]);
	}
	fprintf(fp, "\n");
	free(aux);
}

void cuda_print_double(FILE * fp, double *value)
{
	double aux;
	if(NO_DEBUG_ALL)
		return;
	cudaMemcpy(&aux, value, sizeof(double), cudaMemcpyDeviceToHost);
	fprintf(fp, "%lf", aux);
}

void print_layer_status(Cuda_Network *nn, Cuda_Layer_Type ltype, int print_weights)
{
	Cuda_Layer *l = cuda_get_layer(nn, ltype);

	double *aux_outputs = (double*)calloc(1, sizeof(double) * l->n_output + 1);
	cudaMemcpy(aux_outputs, l->outputs, sizeof(double) * l->n_output, cudaMemcpyDeviceToHost);

	fprintf(stderr, "CUDA_Layer %d: \n", ltype);
	for (int o=0; o<l->n_output - 1 ;o++){
		double *aux_weights = NULL;
		if(l->nodes) {
			aux_weights = (double*)calloc(1, sizeof(double) * l->nodes[o].wcount);
			cudaMemcpy(aux_weights, l->nodes[o].weights, sizeof(double) * l->nodes[o].wcount, cudaMemcpyDeviceToHost);
			fprintf(stderr, "CUDA_Node %d: Bias %lf Output %lf(%p) Weights(%p): \n", o, aux_weights[0], aux_outputs[o+1], &l->outputs[o], l->nodes[o].weights);
		} else
			fprintf(stderr, "CUDA_Node %d: Output %lf Weights: \n", o, aux_outputs[o+1]);

		if(print_weights) {
			for (int i=1; i<l->nodes->wcount; i++){
				fprintf(stderr, "%01.06lf ", aux_weights[i]);
			}
			fprintf(stderr, "\n");
		}
		free(aux_weights);
	}
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

//extern "C" int cuda_get_layer_prediction(Cuda_Layer *l)
//{
//	double maxOut = 0;
//	int maxInd = 0;
//
//	for (int i=0; i < l->n_output; i++){
//		if (l->cell[i].output > maxOut){
//			maxOut = l->cell[i].output;
//			maxInd = i;
//		}
//	}
//
//	return maxInd;
//}

/**
 * @brief Creates a layer with default values
 *
 * @param node_count Number of nodes in layer
 * @param weight_count Number of weights per node
 */
Cuda_Layer *cuda_create_layer(int node_count, int weight_count)
{
	cudaError_t err = cudaSuccess;
	Cuda_Layer *layer;
	
	layer = (Cuda_Layer *)calloc(1, sizeof(Cuda_Layer));

	layer->n_output = node_count + 1;
	err = cudaMalloc((void **)&layer->outputs, sizeof(double) * (node_count + 1));
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate input vec (error code %s)!\n", cudaGetErrorString(err));
		return NULL;
	}

	err = cudaMalloc((void **)&layer->err_signal, sizeof(double) * (node_count + 1));
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate input vec (error code %s)!\n", cudaGetErrorString(err));
		return NULL;
	}

	err = cudaMalloc((void **)&layer->err_sum, sizeof(double) * (node_count + 1));
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate input vec (error code %s)!\n", cudaGetErrorString(err));
		return NULL;
	}

	if(weight_count > 0) {
		layer->nodes = (Cuda_Node *)calloc(1, sizeof(Cuda_Node) * node_count);
		for(int i = 0; i < node_count; i++) {
			layer->nodes[i].wcount = weight_count + 1;
			err = cudaMalloc((void **)&layer->nodes[i].weights, sizeof(double) * (weight_count + 1));
			if (err != cudaSuccess) {
				fprintf(stderr, "Failed to allocate input weight (error code %s)!\n", cudaGetErrorString(err));
				//TODO Liberar y salir bien
				return NULL;
			}
		}
	}

	return layer;
}

/**
 * @brief Initializes the NN by creating and copying INTPUT, HIDDEN, OUTPUT data structures into the NN's memory space
 * @param nn A pointer to the NN
 * @param inpCount Number of nodes in the INPUT layer
 * @param hidCount Number of nodes in the HIDDEN layer
 * @param out_count Number of nodes in the OUTPUT layer
 */
int cuda_init_network(Cuda_Network *nn, int in_count, int hid_count, int out_count)
{
	nn->layers = (Cuda_Layer **)calloc(1, sizeof(Cuda_Layer*) * CUDA_LAYER_CANT);

	nn->layers[0] = cuda_create_layer(in_count, 0);
	nn->layers[1] = cuda_create_layer(hid_count, in_count);
	nn->layers[2] = cuda_create_layer(out_count, hid_count);

	nn->streams = (cudaStream_t *)malloc(30 * sizeof(cudaStream_t));
	for (int i = 0; i < 30; i++){
		cudaStreamCreate(&nn->streams[i]);
	}

	return 0;
}

/**
 * @brief Sets the default network parameters (which can be overwritten/changed)
 * @param nn A pointer to the NN
 */
void cuda_set_network_defaults(Cuda_Network *nn)
{
	//TODO Ajustar learning rate segun funcion de activacion
	// Set deffault activation function types
	nn->hid_layer_act_type = CUDA_ACT_SIGMOID;
	nn->out_layer_act_type = CUDA_ACT_SIGMOID;

	nn->learning_rate = CUDA_LEARNING_RATE_SIGMOID;

	cudaMemcpyToSymbol(_cuda_learning_rate, &nn->learning_rate, sizeof(double));
}

/**
 * @brief Returns one of the layers of the network
 * @param nn A pointer to the NN
 * @param ltype Type of layer to be returned (INPUT, HIDDEN, OUTPUT)
 */
Cuda_Layer *cuda_get_layer(Cuda_Network *nn, Cuda_Layer_Type ltype)
{
	Cuda_Layer *l;

	switch (ltype) {
		case CUDA_LAYER_INPUT:
			l = nn->layers[0];
			break;
		case CUDA_LAYER_HIDDEN:
			l = nn->layers[1];
			break;
		case CUDA_LAYER_OUTPUT:
			l = nn->layers[2];
			break;
		default:
			l = NULL;
	}

	return l;
}

int cuda_layer_init_targets(Cuda_Network *nn, Cuda_Layer_Type ltype)
{
	cudaError_t err = cudaSuccess;
	Cuda_Layer *l = cuda_get_layer(nn, ltype);
	double *aux;
	int i;

	for(i = 0; i < 10; i++) {
		aux = (double*)calloc(1, 11 * sizeof(double));
		if(!aux) {
			fprintf(stderr, "Fallo malloc errno %d %s\n", errno, strerror(errno));
			return -1;
		}

		aux[0] = 1.0;
		aux[i+1] = 1.0;

		err = cudaMalloc((void **)&l->targets[i], sizeof(double) * 11);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to allocate input vec (error code %s)!\n", cudaGetErrorString(err));
			return -1;
		}

		// Copio Vector de salida
		err = cudaMemcpy(l->targets[i], aux, sizeof(double) * 11, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			fprintf(stderr, "%s:: Failed to copy input from host to device cell (error code %s)!\n", __func__, cudaGetErrorString(err));
			free(aux);
			return -1;
		}
		free(aux);
	}

	return 0;
}



int cuda_layer_init_outputs(Cuda_Network *nn, Cuda_Layer_Type ltype)
{
	cudaError_t err = cudaSuccess;
	Cuda_Layer *l = cuda_get_layer(nn, ltype);
	double *aux;

	aux = (double*)calloc(1, l->n_output * sizeof(double));
	if(!aux) {
		fprintf(stderr, "Fallo malloc errno %d %s\n", errno, strerror(errno));
		return -1;
	}
	aux[0] = 1.0;

	// Copio Vector de salida
	err = cudaMemcpy(l->outputs, aux, sizeof(double) * l->n_output, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "%s:: Failed to copy input from host to device cell (error code %s)!\n", __func__, cudaGetErrorString(err));
		free(aux);
		return -1;
	}
	free(aux);

	return 0;
}


int cuda_layer_init_bias(Cuda_Network *nn, Cuda_Layer_Type ltype)
{
	cudaError_t err = cudaSuccess;
	Cuda_Layer *l = cuda_get_layer(nn, ltype);
	Cuda_Node *n = NULL;
	double *aux;

	aux = (double*)malloc(l->n_output * sizeof(double));
	if(!aux) {
		fprintf(stderr, "Fallo malloc errno %d %s", errno, strerror(errno));
		return -1;
	}
	srand(time(NULL));
	for(int o = 0; o < l->n_output; o++) {
		aux[o] = rand()/(double)(RAND_MAX);
		//aux[o] = 0.5;
		if(o%2)
			aux[o] = -aux[o];  // make half of the bias weights negative
	}

	// Copio BIAS
	for(int o = 0; o < l->n_output - 1; o++) {
		if(!n)
			n = l->nodes;

		err = cudaMemcpy(&(n->weights[0]), &aux[o], sizeof(double), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			fprintf(stderr, "%s:: Failed to copy input from host to device cell (error code %s)!\n", __func__, cudaGetErrorString(err));
			free(aux);
			return -1;
		}

		n++;
	}
	free(aux);

	return 0;
}

int cuda_layer_init_weights(Cuda_Network *nn, Cuda_Layer_Type ltype)
{
	cudaError_t err = cudaSuccess;
	Cuda_Layer *l = cuda_get_layer(nn, ltype);
	Cuda_Node *n = NULL;
	double *aux;

	srand(time(NULL));
	for(int o = 0; o < l->n_output - 1; o++) {
		if(!n)
			n = l->nodes;

		aux = (double*)malloc((n->wcount-1) * sizeof(double));
		if(!aux) {
			fprintf(stderr, "Fallo malloc errno %d %s", errno, strerror(errno));
			return -1;
		}

		for(int i = 0; i < n->wcount-1; i++){
			aux[i] = 0.7*(rand()/(double)(RAND_MAX));
			//aux[i] = 0.5;
			if(i%2)
				aux[i] = -aux[i];  // make half of the weights negative
		}

		// Copio weights
		err = cudaMemcpy(&(n->weights[1]), aux, (n->wcount - 1) * sizeof(double), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			fprintf(stderr, "%s:: Failed to copy input from host to device cell (error code %s)!\n", __func__, cudaGetErrorString(err));
			return -1;
		}

		n++;
	}
	free(aux);

	//print_layer_status(nn,ltype);

	return 0;
}

/**
 * @brief Creates a dynamically-sized, 3-layer (INTPUT, HIDDEN, OUTPUT) neural network
 * @param in_count Number of nodes in the INPUT layer
 * @param hid_count Number of nodes in the HIDDEN layer
 * @param out_count Number of nodes in the OUTPUT layer
 */
Cuda_Network *cuda_create_network(int in_count, int hid_count, int out_count)
{
	int ret;
	//Pido memoria para la red
	Cuda_Network *nn = (Cuda_Network*)calloc(1, sizeof(Cuda_Network));

	// Initialize the network by creating the INPUT, HIDDEN and OUTPUT layer inside of it
	cuda_init_network(nn, in_count, hid_count, out_count);

	// Setting defaults
	cuda_set_network_defaults(nn);

	cuda_init_super_input(nn);

	cuda_layer_init_targets(nn, CUDA_LAYER_HIDDEN);
	cuda_layer_init_targets(nn, CUDA_LAYER_OUTPUT);

	// Init connection bias with random values
	ret = cuda_layer_init_bias(nn, CUDA_LAYER_HIDDEN);
	if(ret) goto _create_network_exit_error;
	ret = cuda_layer_init_bias(nn, CUDA_LAYER_OUTPUT);
	if(ret) goto _create_network_exit_error;

	// Init connection weights with random values
	ret = cuda_layer_init_weights(nn, CUDA_LAYER_HIDDEN);
	if(ret) goto _create_network_exit_error;
	ret = cuda_layer_init_weights(nn, CUDA_LAYER_OUTPUT);
	if(ret) goto _create_network_exit_error;

	// Init connection bias with random values
	ret = cuda_layer_init_outputs(nn, CUDA_LAYER_INPUT);
	if(ret) goto _create_network_exit_error;
	ret = cuda_layer_init_outputs(nn, CUDA_LAYER_HIDDEN);
	if(ret) goto _create_network_exit_error;
	ret = cuda_layer_init_outputs(nn, CUDA_LAYER_OUTPUT);
	if(ret) goto _create_network_exit_error;

	return nn;

_create_network_exit_error:
	fprintf(stderr, "Error en la inicializacion de la red\n");
	return NULL;
}

/**
 * @brief Back propagates network error to hidden layer
 * @param nn A pointer to the NN
 * @param targetClassification Correct classification (=label) of the input stream
 */
int cuda_backpropagate_hidden_layer(Cuda_Network *nn, int target_class)
{
	static double *super_weight_vector = NULL;
	cudaError_t err = cudaSuccess;
	int n;
	int blocks_per_grid;
	Cuda_Layer *ol, *hl, *il;

	il = cuda_get_layer(nn, CUDA_LAYER_INPUT);
	hl = cuda_get_layer(nn, CUDA_LAYER_HIDDEN);
	ol = cuda_get_layer(nn, CUDA_LAYER_OUTPUT);

	if(!super_weight_vector) {
		err = cudaMalloc((void **)&super_weight_vector, sizeof(double) * (ol->n_output - 1) * ol->nodes[0].wcount);
		if(err != cudaSuccess) {
			fprintf(stderr, "Failed to allocate device vector super weight (error code %s)!\n", cudaGetErrorString(err));
			return -1;
		}
	}
	for(int i = 1; i < ol->n_output; i++) {
		err = cudaMemcpy(super_weight_vector + ((i-1)*ol->nodes[0].wcount), ol->nodes[i-1].weights, sizeof(double) * ol->nodes[i-1].wcount, cudaMemcpyDeviceToDevice);
	}
	cuda_print_vector(stderr, "SUPER WEIGHT VECTOR", super_weight_vector, (ol->n_output - 1) * ol->nodes[0].wcount);

	//Llamar a kernel para obtener signal de error y update de weights
	//Internamente saltea el primer elemento
	n = ol->n_output;
	blocks_per_grid = MIN(10, (n+THREAD_PER_BLOCK-1)/THREAD_PER_BLOCK);
	vectorGetErrSignal<<<blocks_per_grid, THREAD_PER_BLOCK>>>(ol->targets[target_class], ol->outputs, ol->err_signal, n, 0, 0);
	//cuda_print_vector(stderr, "ERR SIGg", err_signal, n);
	//Ya arme la err_signal

	n = (ol->n_output - 1) * ol->nodes[0].wcount;
	blocks_per_grid = MIN(10, (n+THREAD_PER_BLOCK-1)/THREAD_PER_BLOCK);
	vectorGetErrorSum<<<blocks_per_grid, THREAD_PER_BLOCK>>>(ol->err_signal, super_weight_vector, hl->err_sum, n, hl->n_output, ol->n_output, 0);
	//cuda_print_vector(stderr, "SUM ERROR", dev_buf, hl->n_output);

	n = hl->n_output;
	blocks_per_grid = MIN(10, (n+THREAD_PER_BLOCK-1)/THREAD_PER_BLOCK);
	vectorGetErrSignal<<<blocks_per_grid, THREAD_PER_BLOCK>>>(hl->err_sum, hl->outputs, hl->err_signal, n, 1, 0);
	//cuda_print_vector(stderr, "HID ERR SIGNAL", hid_err_signal, hl->n_output);
	//Update de weights con hid_err_signal
	//fprintf(stderr, "++++ AFTER ERRSIGNAL\n");
	n = il->n_output;
	blocks_per_grid = MIN(10, (n+THREAD_PER_BLOCK-1)/THREAD_PER_BLOCK);
	for(int i = 1; i < hl->n_output; i++) {
		vectorUpdateWeights<<<blocks_per_grid, THREAD_PER_BLOCK, 0, nn->streams[i]>>>(hl->nodes[i-1].weights, il->outputs, &hl->err_signal[i], n, 0);
		//print_layer_status(nn, CUDA_LAYER_HIDDEN, 1);
	}

	//cuda_free(super_weight_vector, -1);

	return 0;
}

/**
 * @brief Back propagates network error in output layer
 * @param nn A pointer to the NN
 * @param targetClassification Correct classification (=label) of the input stream
 */
int cuda_backpropagate_output_layer(Cuda_Network *nn, int target_class)
{
	int n;
	int blocks_per_grid;
	Cuda_Layer *ol, *hl;

	hl = cuda_get_layer(nn, CUDA_LAYER_HIDDEN);
	ol = cuda_get_layer(nn, CUDA_LAYER_OUTPUT);

	n = ol->n_output;
	blocks_per_grid = MIN(10, (n+THREAD_PER_BLOCK-1)/THREAD_PER_BLOCK);

	//Llamar a kernel para obtener signal de error y update de weights
	//Internamente saltea el primer elemento
	vectorGetErrSignal<<<blocks_per_grid, THREAD_PER_BLOCK>>>(ol->targets[target_class], ol->outputs, hl->err_signal, n, 0, 0);
	cuda_print_vector(stderr, "ERROR VECTOR", hl->err_signal, n);

	n = hl->n_output;
	blocks_per_grid = MIN(10, (n+THREAD_PER_BLOCK-1)/THREAD_PER_BLOCK);

	//TODO Ver de actualizar todo.. vectorizar
	for(int i = 1; i < ol->n_output; i++) {
		vectorUpdateWeights<<<1, 64, 0, nn->streams[i]>>>(ol->nodes[i-1].weights, hl->outputs, &hl->err_signal[i], n, 0);
	}

	return 0;
}

/**
 * @brief Back propagates network error from output layer to hidden layer
 * @param nn A pointer to the NN
 * @param targetClassification Correct classification (=label) of the input stream
 */
void cuda_backpropagate_network(Cuda_Network *nn, int target_class)
{
	//fprintf(stderr, "----CUDA Pre backpropagate!\n");
	//print_layer_status(nn, CUDA_LAYER_OUTPUT, 1);
	cuda_backpropagate_output_layer(nn, target_class);
	//fprintf(stderr, "----CUDA Luego de backpropagate!\n");
	//print_layer_status(nn, CUDA_LAYER_OUTPUT, 1);

	//fprintf(stderr, "----CUDA Pre backpropagate HIDDEN!\n");
	//print_layer_status(nn, CUDA_LAYER_HIDDEN, 1);
	cuda_backpropagate_hidden_layer(nn, target_class);
	//fprintf(stderr, "----CUDA Luego de backpropagate HIDDEN!\n");
	//print_layer_status(nn, CUDA_LAYER_HIDDEN, 1);
}

/**
 * @brief Performs an activiation function (as defined in the NN's defaults) to a specified node
 * @param nn A pointer to the NN
 * @param ltype Type of layer (INPUT, HIDDEN, OUTPUT)
 * @param id Sequential id of the node that is to be calculated
 */
void cuda_activate_node(Cuda_Network *nn, Cuda_Layer_Type ltype)
{
	Cuda_Layer *l = cuda_get_layer(nn, ltype);
	int n;
	int blocks_per_grid;

	n = l->n_output;
	blocks_per_grid = MIN(10, (n+THREAD_PER_BLOCK-1)/THREAD_PER_BLOCK);

	Cuda_Act_Func_Type actFct;

	if (ltype == CUDA_LAYER_HIDDEN)
		actFct = nn->hid_layer_act_type;
	else 
		actFct = nn->out_layer_act_type;

	if(actFct == CUDA_ACT_TANH)
		fprintf(stderr, "No se implemento TANH, se fuerza sigmoide");

	sigmoid_kernel<<<blocks_per_grid, THREAD_PER_BLOCK>>>(l->outputs, l->outputs, l->n_output);
}

/**
 * @brief Calculates the output value of a specified node by multiplying all its weights with the previous layer's outputs
 * @param nn A pointer to the NN
 * @param ltype Type of layer (INPUT, HIDDEN, OUTPUT)
 */

int cuda_calc_node_output(Cuda_Network *nn, Cuda_Layer_Type ltype)
{
	int n;
	int blocks_per_grid;
	Cuda_Layer *prev_l, *cur_l;
	//cudaError_t err = cudaSuccess;

	switch (ltype) {
		case CUDA_LAYER_INPUT:
			fprintf(stderr, "Se pidio calcular output de input layer");
			return -1;
		case CUDA_LAYER_HIDDEN:
			prev_l = nn->layers[0];
			cur_l = nn->layers[1];
			break;
		case CUDA_LAYER_OUTPUT:
			prev_l = nn->layers[1];
			cur_l = nn->layers[2];
			break;
		default:
			fprintf(stderr, "Layer invalida! %d", ltype);
			return -1;
	}

	n = prev_l->n_output;
	blocks_per_grid = MIN(10, (n+THREAD_PER_BLOCK-1)/THREAD_PER_BLOCK);
	if(blocks_per_grid > 1) {
		fprintf(stderr, "VOY A FALLLARRRR!\n");
		return -1;
	}

//	fprintf(stderr, "A calcular output type %d:\n", ltype);
//	if(ltype == CUDA_LAYER_HIDDEN)
//		cuda_print_vector(stdout, "IL OUTPUTS", prev_l->outputs, 785);
//	print_layer_status(nn,ltype, 1);

	for (int i = 0; i < cur_l->n_output - 1; i++){
		//vectorDotProduct<<<10, 100>>>(prev_l->outputs, cur_l->nodes[i].weights, &(cur_l->outputs[i+1]), n, 0); //Dio mejores resultados
		//vectorDotProduct<<<blocks_per_grid, THREAD_PER_BLOCK>>>(prev_l->outputs, cur_l->nodes[i].weights, &(cur_l->outputs[i+1]), n, 0);
		//vectorDotProduct<<<10, THREAD_PER_BLOCK/10, 0, nn->streams[i]>>>(prev_l->outputs, cur_l->nodes[i].weights, &(cur_l->outputs[i+1]), n, 0); //hay que sumar los outs.. 154, 214.. no vale la pena y no anda
		vectorDotProduct<<<blocks_per_grid, THREAD_PER_BLOCK, 0, nn->streams[i]>>>(prev_l->outputs, cur_l->nodes[i].weights, &(cur_l->outputs[i+1]), n, 0); //Funciona 156, 211

		//cu_dot<THREAD_PER_BLOCK> <<<blocks_per_grid, THREAD_PER_BLOCK, THREAD_PER_BLOCK * sizeof(double)>>>(prev_l->outputs, cur_l->nodes[i].weights, &(cur_l->outputs[i+1]), n);

		//cudaDeviceSynchronize();
		//fprintf(stderr, "Listo: outputs %p: ", prev_l->outputs, cur_l->nodes[i].weights);
		//cuda_print_double(stderr, &(prev_l->outputs[0]));
		//fprintf(stderr, ".. weights %p: ", prev_l->outputs, cur_l->nodes[i].weights);
		//cuda_print_double(stderr, &(cur_l->nodes[i].weights[0]));
		//fprintf(stderr, ".. Output: ", prev_l->outputs, cur_l->nodes[i].weights);
		//cuda_print_double(stderr, &(cur_l->outputs[i+1]));
		//fprintf(stderr, "\n");
	}
	//fprintf(stdout, "Listo_outputs\n");
	//free(streams);

	return 0;
}

/**
 * @brief Calculates the output values of a given NN layer
 * @param nn A pointer to the NN
 * @param ltype Type of layer (INPUT, HIDDEN, OUTPUT)
 */
void cuda_calc_layer(Cuda_Network *nn, Cuda_Layer_Type ltype)
{
	Cuda_Layer *l = cuda_get_layer(nn, ltype);

//	fprintf(stderr, "Cuda PRE Calculando... %d: inp!\n", ltype);
//	print_layer_status(nn,CUDA_LAYER_INPUT, 0);
//	fprintf(stderr, "Cuda PRE Calculando... %d: hid!\n", ltype);
//	print_layer_status(nn,ltype, 1);
	cuda_calc_node_output(nn, ltype);

	//fprintf(stderr, "Cuda_Calculando... %d: OUTPUT!\n", ltype);
	//print_layer_status(nn,ltype, 1);

	cudaDeviceSynchronize(); //Sincronizo para tener el output bien

	cuda_activate_node(nn, ltype);
	//fprintf(stderr, "Cuda_Calculando... %d: ACTIVATED!\n", ltype);
	//print_layer_status(nn,ltype, 1);
}

/**
 * @brief Feeds input layer values forward to hidden to output layer (calculation and activation fct)
 * @param nn A pointer to the NN
 */
void cuda_feed_forward_network(Cuda_Network *nn)
{
	//fprintf(stderr, "A calcular layer HIDDEN %d\n", CUDA_LAYER_HIDDEN);
	cuda_calc_layer(nn, CUDA_LAYER_HIDDEN);
	//print_layer_status(nn, CUDA_LAYER_HIDDEN, 1);

	//fprintf(stderr, "A calcular layer OUTPUT %d\n", CUDA_LAYER_OUTPUT);
	cuda_calc_layer(nn, CUDA_LAYER_OUTPUT);
	//print_layer_status(nn, CUDA_LAYER_OUTPUT, 1);
}


/**
 * @brief Feeds some Vector data into the INPUT layer of the NN
 * @param nn A pointer to the NN
 * @param v A pointer to a vector
 */
int cuda_feed_input(Cuda_Network *nn, Vector *v)
{
	cudaError_t err = cudaSuccess;
	Cuda_Layer *il;
	il = nn->layers[0]; //Layer 0 es la input

	//uint64_t ts = cu_get_time_usec();
	err = cudaMemcpy(&(il->outputs[1]), v->vals, v->size * sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy input from host to device cell (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}
	//printf("%s:: time to copy %llu\n", __func__, cu_get_time_usec() - ts);

	//printInput<<<BLOCK_PER_GRID, THREAD_PER_BLOCK>>>(c->input, c->n_inputs);

	return 0;
}

/**
 * @brief Returns the network's classification using the ID of teh node with the hightest output
 * @param nn A pointer to the NN
 */
int cuda_get_network_classification(Cuda_Network *nn)
{
	cudaError_t err = cudaSuccess;
	int n, blocks_per_grid;
	//double *dev_max = 0, host_max;
	int *dev_ind = 0, host_ind;
	Cuda_Layer *ol = cuda_get_layer(nn, CUDA_LAYER_OUTPUT);

	//err = cudaMalloc((void **)&dev_max, sizeof(double));
	//if(err != cudaSuccess) {
	//	fprintf(stderr, "Failed to allocate device vector dev_max (error code %s)!\n", cudaGetErrorString(err));
	//	return -1;
	//}

	cudaMalloc((void **)&dev_ind, sizeof(int));
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector dev_ind (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}

	n = ol->n_output;
	blocks_per_grid = MIN(10, (n+THREAD_PER_BLOCK-1)/THREAD_PER_BLOCK);
	naiveGetMax<<<blocks_per_grid, n>>>(ol->outputs, ol->n_output, NULL, dev_ind, 0);
	//cudaDeviceSynchronize();
	//cuda_print_vector(stderr, "OUTPUT", ol->outputs, (ol->n_output));

	//cudaMemcpy(&host_max, dev_max, sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(&host_ind, dev_ind, sizeof(int), cudaMemcpyDeviceToHost);

	//fprintf(stderr, "ASDASDDSASADASDASDSADDSA %f %d\n", host_max, host_ind);

	//cuda_free(dev_max, -1);
	cuda_free(dev_ind, -1);

	return host_ind;
}

int cuda_init_super_input(Cuda_Network *nn)
{
	cudaError_t err = cudaSuccess;
	err = cudaMalloc((void **)&nn->super_input, sizeof(double) * (784+1) * 60000);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector dev_max (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}

	return 0;
}

int cuda_copy_to_super_input(Cuda_Network *nn, double *input_data)
{
	cudaError_t err = cudaSuccess;

	err = cudaMemcpy(nn->super_input, input_data, sizeof(double) * 60000 * (784 + 1), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy input from host to device cell (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}
	//cuda_print_vector(stdout, "SUPER INP", nn->super_input, 3);

	return 0;
}

/**
 * @brief Feeds some Vector data into the INPUT layer of the NN
 * @param nn A pointer to the NN
 * @param v A pointer to a vector
 */
int cuda_feed_input_from_super_input(Cuda_Network *nn, int i)
{
	Cuda_Layer *il = cuda_get_layer(nn, CUDA_LAYER_INPUT);

	//XXX Leak del primer output!!
	il->outputs = nn->super_input + (i * (784+1));

	//cuda_print_vector(stdout, "SUPER INP", nn->super_input + (i * (784+1)), 785);

	return 0;
}


