/**
 * @file main.c
 *
 * @mainpage MNIST 1-Layer Neural Network
 *
 * @brief Main characteristics: Only 1 layer (= input layer), no hidden layer.  Feed-forward only.
 * No Sigmoid activation function. No back propagation.\n
 *
 * @details Learning is achieved simply by incrementally updating the connection weights based on the comparison
 * with the desired target output (supervised learning).\n
 *
 * Its performance (success rate) of 85% is far off the state-of-the-art techniques (surprise, surprise) 
 * but close the Yann Lecun's 88% when using only a linear classifier.
 *
 * @see [Simple 1-Layer Neural Network for MNIST Handwriting Recognition](http://mmlind.github.io/Simple_1-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/)
 * @see http://yann.lecun.com/exdb/mnist/
 * @version [Github Project Page](http://github.com/mmlind/mnist-1lnn/)
 * @author [Matt Lind](http://mmlind.github.io)
 * @date July 2015
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#include "screen.h"
#include "mnist-utils.h"
#include "mnist-stats.h"
#include "3lnn.h"

#include "cuda_utils.h"

#include "utils_queue.h"

Utils_Queue *queue;

void *read_thread(void *args);

/**
 * @details Trains a layer by looping through and training its cells
 * @param l A pointer to the layer that is to be training
 */
void trainNetwork(Network *nn, Cuda_Network *cu_nn)
{
	int ret;
	int img_count = 0;

	int errCount = 0;
	int cu_errCount = 0;

	while(img_count < MNIST_MAX_TRAINING_IMAGES) {
		MNIST_Packet *pkt;

		pkt = utils_queue_get(queue);
		if(!pkt) {
			usleep(1000);
			continue;
		}
		img_count ++;


		//fprintf(stderr, "==== IMAGEN NUMERO %d\n", img_count);

		feedInput(nn, pkt->vec);
		//cuda_feed_input(cu_nn, pkt->vec);

		feedForwardNetwork(nn);
		//cuda_feed_forward_network(cu_nn);

		backPropagateNetwork(nn, pkt->label);
		//cuda_backpropagate_network(cu_nn, pkt->label);

		//displayImage(pkt->img, 6,6);

		int classification = getNetworkClassification(nn);
		if (classification != pkt->label) errCount++;

		//int cu_predictedNum;
		//cu_predictedNum = cuda_get_network_classification(cu_nn);
		//if(cu_predictedNum != pkt->label) cu_errCount++;

		//printf("\n      Voy por: %d      Hay   : %d ",img_count, utils_queue_get_count(queue));
		//printf("\n      Prediction: %d   Actual: %d ",classification, pkt->label);
		//printf("\n cuda Prediction: %d   Actual: %d ",cu_predictedNum, pkt->label);
		//getchar();

		displayTrainingProgress(img_count, errCount, 3,5);
		//displayTrainingProgress(img_count, cu_errCount, 13,5);



		free(pkt->vec);
		free(pkt);
	}
	printf("\n TERMINEE\n");
}

/**
 * @details Tests a layer by looping through and testing its cells
 * Exactly the same as TrainLayer() but WITHOUT LEARNING.
 * @param l A pointer to the layer that is to be training
 */

/**
 * @brief Testing the trained network by processing the MNIST testing set WITHOUT updating weights
 * @param nn A pointer to the NN
 */

void testNetwork(Network *nn){

	// open MNIST files
	FILE *imageFile, *labelFile;
	imageFile = openMNISTImageFile(MNIST_TESTING_SET_IMAGE_FILE_NAME);
	labelFile = openMNISTLabelFile(MNIST_TESTING_SET_LABEL_FILE_NAME);

	int errCount = 0;

	// Loop through all images in the file
	for (int imgCount=0; imgCount<MNIST_MAX_TESTING_IMAGES; imgCount++){

		// Reading next image and its corresponding label
		MNIST_Image img = getImage(imageFile);
		MNIST_Label lbl = getLabel(labelFile);

		// Convert the MNIST image to a standardized vector format and feed into the network
		Vector *inpVector = getVectorFromImage(&img);
		feedInput(nn, inpVector);

		// Feed forward all layers (from input to hidden to output) calculating all nodes' output
		feedForwardNetwork(nn);

		// Classify image by choosing output cell with highest output
		int classification = getNetworkClassification(nn);
		if (classification!=lbl) errCount++;

		// Display progress during testing
		displayTestingProgress(imgCount, errCount, 5,5);
		//        displayImage(&img, lbl, classification, 7,6);

	}

	// Close files
	fclose(imageFile);
	fclose(labelFile);

}

void *read_thread(void *args)
{
	int reading = 1;
	int img_count = 0;
	FILE *imageFile, *labelFile;
	imageFile = openMNISTImageFile(MNIST_TRAINING_SET_IMAGE_FILE_NAME);
	labelFile = openMNISTLabelFile(MNIST_TRAINING_SET_LABEL_FILE_NAME);

	while(img_count < MNIST_MAX_TRAINING_IMAGES) {
		MNIST_Packet *pkt;
		if(utils_queue_get_count(queue) > 10000 && reading) {
			reading = 0;
		}
		if(utils_queue_get_count(queue) < 50  && reading == 0) {
			reading = 1;
		}
		if(!reading) {
			usleep(1000);
			continue;
		}
		img_count ++;

		MNIST_Image img = getImage(imageFile);
		MNIST_Label lbl = getLabel(labelFile);

		pkt = calloc(1, sizeof(MNIST_Packet));
		pkt->vec = getVectorFromImage(&img);
		pkt->label = lbl;

		utils_queue_insert(queue, (void *)pkt);
	}

	printf("\n Termine de leer.. meti %d \n", img_count);
	fclose(imageFile);
	fclose(labelFile);
}

/**
 * @details Main function to run MNIST-1LNN
 */

int main(int argc, const char * argv[])
{
	// remember the time in order to calculate processing time at the end
	time_t startTime = time(NULL);
	pthread_t read_th;

	// clear screen of terminal window
	clearScreen();
	printf("    MNIST-1LNN: a simple 1-layer neural network processing the MNIST handwriting images\n");

	queue = utils_queue_alloc();

	//Inicio thread de lectura
	pthread_create(&read_th, NULL, read_thread, NULL);

	//Creo la red
	Network *nn = createNetwork(MNIST_IMG_HEIGHT * MNIST_IMG_WIDTH, 20, 10);
	Cuda_Network *cu_nn = cuda_create_network(MNIST_IMG_HEIGHT * MNIST_IMG_WIDTH, 20, 10);

	// inicializacion cuda
	//Cuda_Layer cuda_outputLayer;
//	if(cuda_init_layer(&cuda_outputLayer, NUMBER_OF_INPUT_CELLS, NUMBER_OF_OUTPUT_CELLS)) {
//		fprintf(stderr, "Error inicializando cuda layer\n");
//		return -1;
//	}

	trainNetwork(nn, cu_nn);

	testNetwork(nn);

	locateCursor(38, 5);

	// Calculate and print the program's total execution time
	time_t endTime = time(NULL);
	double executionTime = difftime(endTime, startTime);
	printf("\n    DONE! Total execution time: %.1f sec\n\n",executionTime);

	pthread_join(read_th, NULL);

	return 0;
}


