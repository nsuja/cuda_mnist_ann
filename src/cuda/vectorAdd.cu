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

#define min(x,y) (x>y?x:y)
#define N 784

#define THREAD_PER_BLOCK 16

//smallest multiple of threadsPerBlock that is greater than or equal to N
#define BLOCK_PER_GRID min(32 , (N+THREAD_PER_BLOCK-1) / THREAD_PER_BLOCK )

#define CUDA_LAYER_CANT (3)

#define CUDA_LEARNING_RATE_SIGMOID (0.004) //91.5%
#define CUDA_LEARNING_RATE_TANH (0.2) //78%

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
	double *bias;
	double *outputs;
	Cuda_Node *nodes;
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

void print_layer_status(Cuda_Network *nn, Cuda_Layer_Type ltype)
{
	Cuda_Layer *l = cuda_get_layer(nn, ltype);

	double *aux_bias = (double*)calloc(1, sizeof(double) * l->n_output);
	cudaMemcpy(aux_bias, l->bias, sizeof(double) * l->n_output, cudaMemcpyDeviceToHost);

	fprintf(stderr, "CUDA_Layer %d: \n");
	for (int o=0; o<l->n_output;o++){
		double *aux_weights = (double*)calloc(1, sizeof(double) * l->nodes[o].wcount);
		cudaMemcpy(aux_weights, l->nodes[o].weights, sizeof(double) * l->nodes[o].wcount, cudaMemcpyDeviceToHost);

		fprintf(stderr, "CUDA_Node %d: Bias %lf Weights: \n", o, aux_bias[o]);
		for (int i=0; i<l->nodes->wcount; i++){
			fprintf(stderr, "%1.6lf ", aux_weights[i]);
		}
		fprintf(stderr, "\n");
		free(aux_weights);
	}
	free(aux_bias);
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

//extern "C" int cuda_train_cell(Cuda_Layer *l, int n_cell, MNIST_Image *img, int target)
//{
//	int ret;
//	Cuda_Cell *c;
//	c = &l->cell[n_cell];
//	cudaDeviceSynchronize();
//	ret = cuda_set_cell_input(c, img);
//	if(ret) {
//		return -1;
//	}
//	cuda_calc_cell_output(c);
//
//	// learning (by updating the weights)
//	double err = get_cell_error(c, target);
//	update_cell_weights(c, err);
//	cudaDeviceSynchronize();
//	return 0;
//}

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
	layer->n_output = node_count;
	layer->nodes = (Cuda_Node *)calloc(1, sizeof(Cuda_Node) * node_count);

	err = cudaMalloc((void **)&layer->outputs, sizeof(double) * node_count);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate input vec (error code %s)!\n", cudaGetErrorString(err));
		return NULL;
	}

	err = cudaMalloc((void **)&layer->bias, sizeof(double) * node_count);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate input vec (error code %s)!\n", cudaGetErrorString(err));
		return NULL;
	}

	for(int i = 0; i < node_count; i++) {
		layer->nodes[i].wcount = weight_count;
		err = cudaMalloc((void **)&layer->nodes[i].weights, sizeof(double) * weight_count);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to allocate input weight (error code %s)!\n", cudaGetErrorString(err));
			//TODO Liberar y salir bien
			return NULL;
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

/**
 * @brief Initializes a layer's weights with random values
 * @param nn A pointer to the NN
 * @param ltype Defining what layer to initialize
 */
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
		if(!n)
			n = l->nodes;

		aux[o] = rand()/(double)(RAND_MAX);
		if(o%2)
			aux[o] = -aux[o];  // make half of the bias weights negative
	}

	// Copio BIAS
	err = cudaMemcpy(l->bias, aux, l->n_output * sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy input from host to device cell (error code %s)!\n", cudaGetErrorString(err));
		free(aux);
		return -1;
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
	for(int o = 0; o < l->n_output; o++) {
		if(!n)
			n = l->nodes;

		aux = (double*)malloc(n->wcount * sizeof(double));
		if(!aux) {
			fprintf(stderr, "Fallo malloc errno %d %s", errno, strerror(errno));
			return -1;
		}

		for(int i = 0; i < n->wcount; i++){
			aux[i] = 0.7*(rand()/(double)(RAND_MAX));
			if(i%2)
				aux[i] = -aux[i];  // make half of the weights negative
		}

		// Copio weights
		err = cudaMemcpy(n->weights, aux, n->wcount * sizeof(double), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to copy input from host to device cell (error code %s)!\n", cudaGetErrorString(err));
			return -1;
		}

		n++;
	}
	free(aux);

	print_layer_status(nn,ltype);

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
	Cuda_Network *nn = (Cuda_Network*)calloc(1, sizeof(Cuda_Network) + i_layer_size + h_layer_size + o_layer_size);

	// Set/remember byte sizes of each component of the network
	nn->i_node_size     = i_node_size;
	nn->i_layer_size    = i_layer_size;
	nn->h_node_size     = h_node_size;
	nn->h_layer_size    = h_layer_size;
	nn->o_node_size     = o_node_size;
	nn->o_layer_size    = o_layer_size;

	// Initialize the network by creating the INPUT, HIDDEN and OUTPUT layer inside of it
	cuda_init_network(nn, in_count, hid_count, out_count);

	// Setting defaults
	cuda_set_network_defaults(nn);

	// Init connection bias with random values
	cuda_layer_init_bias(nn, CUDA_LAYER_HIDDEN);
	cuda_layer_init_bias(nn, CUDA_LAYER_OUTPUT);

	// Init connection weights with random values
	cuda_layer_init_weights(nn, CUDA_LAYER_HIDDEN);
	cuda_layer_init_weights(nn, CUDA_LAYER_OUTPUT);

	return nn;
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

	sigmoid_kernel<<<BLOCK_PER_GRID, THREAD_PER_BLOCK>>>(l->outputs, l->outputs, l->n_output);

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
	Cuda_Layer *prev_l, *cur_l;
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
	//FIXME Ver si esto funciona
	cur_l->outputs = cur_l->bias;

	double *V3_D;
	double sum = 0;

	//cur_l->output=0;

	//printf("%d %d\n", THREAD_PER_BLOCK, BLOCK_PER_GRID);
	//V3_H = (double *)calloc(1, sizeof(double) * BLOCK_PER_GRID);
	cudaMalloc((void **)&V3_D, BLOCK_PER_GRID*sizeof(double));

	cudaDeviceSynchronize();
	for (int i = 0; i < cur_l->n_output; i++){
		vectorDotProduct<<<BLOCK_PER_GRID, THREAD_PER_BLOCK>>>(prev_l->outputs, cur_l->nodes[i].weights, V3_D);
		cudaDeviceSynchronize();
		//cudaMemcpy(V3_H, V3_D, BLOCK_PER_GRID*sizeof(double), cudaMemcpyDeviceToHost);

		for(int j = 0; j < BLOCK_PER_GRID; j++ )
			cur_l->outputs[i] += V3_D[j];

		//c->output = sum / c->n_inputs; // normalize output (0-1)
		fprintf(stderr, "%s:: output %f %f\n", __func__, sum, cur_l->outputs[i]);
	}

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

	err = cudaMemcpy(il->outputs, v->vals, v->size * sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy input from host to device cell (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}

	//printInput<<<BLOCK_PER_GRID, THREAD_PER_BLOCK>>>(c->input, c->n_inputs);

	return 0;
}
