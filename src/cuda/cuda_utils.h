#pragma once

#ifdef __CUDACC__
extern "C" {
#endif

typedef struct Cuda_Cell Cuda_Cell;
typedef struct Cuda_Layer Cuda_Layer;
typedef struct Cuda_Vector Cuda_Vector;

/**
 * @brief The single (output) layer of this network (a layer is number cells)
 */
struct Cuda_Layer{
    Cuda_Cell *cell;
};

/**
 * @brief Data structure containing defined number of integer values (the output vector contains values for 0-9)
 */
struct Cuda_Vector{
    int *val;
};

int copy_to_cuda(uint8_t *buf, int size);
//int cuda_init_layer(Cuda_Layer *l, int n_input_cells, int n_output_cells);

#ifdef __CUDACC__
}
#endif
