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

#define MIN(x,y) (x < y ? x:y)

#define THREAD_PER_BLOCK (1024)
#define MAX_BLOCKS (10)

#define CUDA_LAYER_CANT (3)

#define CUDA_LEARNING_RATE_SIGMOID (0.004) //91.5%
#define CUDA_LEARNING_RATE_TANH (0.2) //78%

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
	Cuda_Layer **layers;
};


Cuda_Layer *cuda_get_layer(Cuda_Network *nn, Cuda_Layer_Type ltype);


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

	//printf("Hola desde kernel blockdim %d blockidx %d thidx %d\n", blockDim.x, blockIdx.x, threadIdx.x);
	while ( tid < size ) {
		temp += V1[tid] * V2[tid] ;
		if(log)
			printf("(%d, %d, %d) tid %d .. %f %f temp %f\n", blockDim.x, blockIdx.x, threadIdx.x, tid, V1[tid], V2[tid], temp);
		tid += blockDim.x * gridDim.x;
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

__device__ __forceinline__ double sigmoid (double a)
{
	return 1.0 / (1.0 + exp (-a));
}

__global__ void sigmoid_kernel (const double * __restrict__ src, double * __restrict__ dst, int len)
{
	int stride = gridDim.x * blockDim.x;
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	for (int i = tid; i < len; i += stride) {
		dst[i] = sigmoid (src[i]);
	}
}

void cuda_print_double(FILE * fp, double *value)
{
	double aux;
	cudaMemcpy(&aux, value, sizeof(double), cudaMemcpyDeviceToHost);
	fprintf(fp, "%lf", aux);
}

void print_layer_status(Cuda_Network *nn, Cuda_Layer_Type ltype)
{
	Cuda_Layer *l = cuda_get_layer(nn, ltype);

	double *aux_outputs = (double*)calloc(1, sizeof(double) * l->n_output + 1);
	cudaMemcpy(aux_outputs, l->outputs, sizeof(double) * l->n_output, cudaMemcpyDeviceToHost);

	fprintf(stderr, "CUDA_Layer %d: \n");
	for (int o=0; o<l->n_output - 1 ;o++){
		double *aux_weights = (double*)calloc(1, sizeof(double) * l->nodes[o].wcount);
		cudaMemcpy(aux_weights, l->nodes[o].weights, sizeof(double) * l->nodes[o].wcount, cudaMemcpyDeviceToHost);

		fprintf(stderr, "CUDA_Node %d: Bias %lf Output %lf Weights: \n", o, aux_weights[0], aux_outputs[o+1]);
		if(ltype != CUDA_LAYER_HIDDEN) {
			for (int i=0; i<l->nodes->wcount; i++){
				fprintf(stderr, "%s %1.6lf - ", i == 0 ? "BIAS" : "", aux_weights[i]);
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
	srand(0);
	for(int o = 0; o < l->n_output; o++) {
		aux[o] = rand()/(double)(RAND_MAX);
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

	srand(0);
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
 * @brief Performs an activiation function (as defined in the NN's defaults) to a specified node
 * @param nn A pointer to the NN
 * @param ltype Type of layer (INPUT, HIDDEN, OUTPUT)
 * @param id Sequential id of the node that is to be calculated
 */
void activate_node(Cuda_Network *nn, Cuda_Layer_Type ltype)
{
	Cuda_Layer *l = cuda_get_layer(nn, ltype);
	//Cuda_Node *n = getNode(l, id);

	Cuda_Act_Func_Type actFct;

	if (ltype == CUDA_LAYER_HIDDEN)
		actFct = nn->hid_layer_act_type;
	else 
		actFct = nn->out_layer_act_type;

	if(actFct == CUDA_ACT_TANH)
		fprintf(stderr, "No se implemento TANH, se fuerza sigmoide");

	//sigmoid_kernel<<<BLOCK_PER_GRID, THREAD_PER_BLOCK>>>(l->outputs, l->outputs, l->n_output);

	//if (actFct==TANH)   n->output = tanh(n->output);
	//else n->output = 1 / (1 + (exp((double)-n->output)) );
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
	double *V3_D, *V3_H;
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

	printf("bpg %d tpb %d\n", blocks_per_grid, THREAD_PER_BLOCK);
	V3_H = (double *)calloc(1, sizeof(double) * blocks_per_grid);
	cudaMalloc((void **)&V3_D, blocks_per_grid * sizeof(double));

	fprintf(stderr, "A calcular output type %d\n", ltype);

	cudaDeviceSynchronize();
	for (int i = 0; i < cur_l->n_output - 1; i++){
		fprintf(stderr, "output %d wcount %d prev_l cant %d BLOCK %d\n", i, cur_l->nodes[i].wcount, prev_l->n_output, blocks_per_grid);
		vectorDotProduct<<<blocks_per_grid, THREAD_PER_BLOCK>>>(prev_l->outputs, cur_l->nodes[i].weights, &(cur_l->outputs[i+1]), n, ltype == CUDA_LAYER_HIDDEN);
		cudaDeviceSynchronize();
		fprintf(stdout, "Output: ");
		cuda_print_double(stdout, &(cur_l->outputs[i+1]));
		fprintf(stdout, "\n");
		print_layer_status(nn,ltype);

		//c->output = sum / c->n_inputs; // normalize output (0-1)
		//fprintf(stderr, "%s:: output %f %f\n", __func__, sum, cur_l->outputs[i]);
	}
	free(V3_H);

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

	cuda_calc_node_output(nn, ltype);
	print_layer_status(nn,ltype);
	//cuda_activate_node(nn, ltype);
	//for(int i = 0; i < l->ncount; i++){
	//	cuda_activate_node(nn, ltype, i);
	//}
}

/**
 * @brief Feeds input layer values forward to hidden to output layer (calculation and activation fct)
 * @param nn A pointer to the NN
 */
void cuda_feed_forward_network(Cuda_Network *nn)
{
	cuda_calc_layer(nn, CUDA_LAYER_HIDDEN);
	cuda_calc_layer(nn, CUDA_LAYER_OUTPUT);
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

	err = cudaMemcpy(&(il->outputs[1]), v->vals, v->size * sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy input from host to device cell (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}

	//printInput<<<BLOCK_PER_GRID, THREAD_PER_BLOCK>>>(c->input, c->n_inputs);

	return 0;
}
