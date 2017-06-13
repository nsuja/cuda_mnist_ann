#pragma once

#ifdef __CUDACC__
extern "C" {
#endif

#include "../utils/mnist-utils.h"

typedef struct Cuda_Cell Cuda_Cell;
typedef struct Cuda_Layer Cuda_Layer;
typedef struct Cuda_Vector Cuda_Vector;
typedef struct Cuda_Network Cuda_Network;

int copy_to_cuda(uint8_t *buf, int size);
int cuda_init_layer(Cuda_Layer *l, int n_input_cells, int n_output_cells);
int cuda_train_cell(Cuda_Layer *l, int cell_n, MNIST_Image *img, int target);
int cuda_get_layer_prediction(Cuda_Layer *l);
Cuda_Network *cuda_create_network(int in_count, int hid_count, int out_count);
int cuda_feed_input(Cuda_Network *nn, Vector *v);
void cuda_feed_forward_network(Cuda_Network *nn);
void cuda_backpropagate_network(Cuda_Network *nn, int target_class);

#ifdef __CUDACC__
}
#endif
