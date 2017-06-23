#include <getopt.h>
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
#include "utils_time.h"

Utils_Queue *queue;

int with_cpu = 0;
int with_cuda = 0;
int inform_timestamps = 0;
int verbose = 0;
int max_images = 60000;
int last_cursor_y = 0;

uint8_t *input_data_u8;
double *input_data;
uint8_t *input_labels;

void *read_thread(void *args);

void trainNetwork(Network *nn, Cuda_Network *cu_nn)
{
	int ret;
	int img_count = 0;
	int errCount = 0;
	int cu_errCount = 0;
	uint64_t ts;
	int cu_predictedNum;
	int classification;
	uint64_t ts1, ts2, ts3, ts4, ts5, ts6, ts7;
	ts1 = ts2 = ts3 = ts4 = ts5 = ts6 = ts7 = 0;

	while(img_count < MNIST_MAX_TRAINING_IMAGES) {
		//fprintf(stderr, "==== IMAGEN NUMERO %d\n", img_count);

		if(inform_timestamps)
			ts = get_time_usec();
		if(with_cpu)
			feedInputFixed(nn, &input_data_u8[img_count * (784+1)+1], 784);
		if(inform_timestamps)
			ts1 += get_time_usec() - ts;
		//printf("ts1:: %llu feed_input\n", get_time_usec() - ts);

		if(inform_timestamps)
			ts = get_time_usec();
		if(with_cuda)
			cuda_feed_input_from_super_input(cu_nn, img_count);
		if(inform_timestamps)
			ts2 += get_time_usec() - ts;
		//printf("ts2:: %llu cuda_feed_input\n", get_time_usec() - ts);

		if(inform_timestamps)
			ts = get_time_usec();
		if(with_cpu)
			feedForwardNetwork(nn);
		if(inform_timestamps)
			ts3 += get_time_usec() - ts;
		//printf("ts3:: %llu feed_forward\n", get_time_usec() - ts);

		if(inform_timestamps)
			ts = get_time_usec();
		if(with_cuda)
			cuda_feed_forward_network(cu_nn);
		if(inform_timestamps)
			ts4 += get_time_usec() - ts;
		//printf("ts4:: %llu cuda_feed_forward\n", get_time_usec() - ts);

		if(inform_timestamps)
			ts = get_time_usec();
		if(with_cpu)
			backPropagateNetwork(nn, input_labels[img_count]);
		if(inform_timestamps)
			ts5 += get_time_usec() - ts;
		//printf("ts5:: %llu backpropagation\n", get_time_usec() - ts);

		if(inform_timestamps)
			ts = get_time_usec();
		if(with_cuda)
			cuda_backpropagate_network(cu_nn, input_labels[img_count]);
		if(inform_timestamps)
			ts6 += get_time_usec() - ts;
		//printf("ts6:: %llu cuda backpropagation\n", get_time_usec() - ts);

		if(with_cpu) {
			classification = getNetworkClassification(nn);
			if (classification != input_labels[img_count]) errCount++;
		}

		if(with_cuda) {
			cu_predictedNum = cuda_get_network_classification(cu_nn);
			if(cu_predictedNum != input_labels[img_count]) cu_errCount++;
		}

		if(with_cpu && verbose) {
			displayTrainingProgress(img_count, errCount, 8,1);
		}
		if(with_cuda && verbose) {
			displayTrainingProgress(img_count, cu_errCount, 9,1);
		}

		if(max_images > 0 && img_count - 1 == max_images) {
			return;
		}

		img_count ++;
	}
	if(inform_timestamps) {
		locateCursor(12, 1);
		printf("Timestamps: \n");
		printf("ts1:: %lf %llu %d\n", (double)ts1/(double)img_count, ts1, img_count);
		printf("ts2:: %lf %llu %d\n", (double)ts2/(double)img_count, ts2, img_count);
		printf("ts3:: %lf %llu %d\n", (double)ts3/(double)img_count, ts3, img_count);
		printf("ts4:: %lf %llu %d\n", (double)ts4/(double)img_count, ts4, img_count);
		printf("ts5:: %lf %llu %d\n", (double)ts5/(double)img_count, ts5, img_count);
		printf("ts6:: %lf %llu %d\n", (double)ts6/(double)img_count, ts6, img_count);
	}
}

void testNetwork(Network *nn, Cuda_Network *cu_nn)
{
	FILE *imageFile, *labelFile;
	imageFile = openMNISTImageFile(MNIST_TESTING_SET_IMAGE_FILE_NAME);
	labelFile = openMNISTLabelFile(MNIST_TESTING_SET_LABEL_FILE_NAME);

	int errCount = 0;
	int cu_errCount = 0;
	int classification;
	int cu_predictedNum;

	for (int imgCount=0; imgCount<MNIST_MAX_TESTING_IMAGES; imgCount++){

		MNIST_Image img = getImage(imageFile);
		MNIST_Label lbl = getLabel(labelFile);

		Vector *inpVector = getVectorFromImage(&img);
		if(with_cpu)
			feedInput(nn, inpVector);
		if(with_cuda)
			cuda_feed_input(cu_nn, inpVector);

		if(with_cpu)
			feedForwardNetwork(nn);
		if(with_cuda)
			cuda_feed_forward_network(cu_nn);

		if(with_cpu) {
			classification = getNetworkClassification(nn);
			if (classification!=lbl) errCount++;
		}

		if(with_cuda) {
			cu_predictedNum = cuda_get_network_classification(cu_nn);
			if(cu_predictedNum != lbl) cu_errCount++;
		}

		if(with_cpu)
			displayTestingProgress(imgCount, errCount, 10,1);
		if(with_cuda)
			displayTestingProgress(imgCount, cu_errCount, 11,1);
	}

	fclose(imageFile);
	fclose(labelFile);
}

void *read_data(void)
{
	int reading = 1;
	int img_count = 0;
	FILE *imageFile, *labelFile;
	imageFile = openMNISTImageFile(MNIST_TRAINING_SET_IMAGE_FILE_NAME);
	labelFile = openMNISTLabelFile(MNIST_TRAINING_SET_LABEL_FILE_NAME);

	while(img_count < MNIST_MAX_TRAINING_IMAGES) {
		MNIST_Image img = getImage(imageFile);
		MNIST_Label lbl = getLabel(labelFile);

		input_data[img_count * (28 * 28 + 1)] = 1; //BIAS
		input_data_u8[img_count * (28 * 28 + 1)] = 1; //BIAS
		loadInputData(&img, &input_data[img_count * (28 * 28 + 1) + 1]);
		loadInputDataU8(&img, &input_data_u8[img_count * (28 * 28 + 1) + 1]);
		input_labels[img_count] = (uint8_t)lbl;

		img_count ++;
	}

	//printf("\n Termine de leer.. meti %d \n", img_count);
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

	//printf("\n Termine de leer.. meti %d \n", img_count);
	fclose(imageFile);
	fclose(labelFile);
}

void print_help(const char *argv[])
{
	fprintf(stderr,
			"Uso: %s -m [MODO] [OPCIONES]\n"
			"Entrena y verifica el funcionamiento de la red neuronal de 3 layers\n"
			"\n"
			"Obligatorias\n"
			"  -m            Modo de funcionamiento: \n"
			"                  -m 1 : CUDA \n"
			"                  -m 2 : CPU \n"
			"                  -m 3 : CUDA + CPU \n"
			"Opcional\n"
			"  -h            Imprimir este mensaje\n"
			"  -v            Verbose (default: 0)\n"
			"  -t            Tomar timestamps (default: 0)\n"
			"  -l            Limite de imagenes de entrenamiento (default: 60000)\n"
			"Ejemplos\n"
			"  %s -m 1 -vt         Ejecuta la red neuronal solo CUDA, modo verbose e informe de timestamps\n"
			"  %s -m 3 -t          Ejecuta la red neuronal CUDA + CPU, con informe de timestamps\n"
			"  %s -m 2 -l 10000    Ejecuta la red neuronal CUDA y entrena con 10000 imagenes\n"
			, argv[0], argv[0], argv[0], argv[0]);
}

int main(int argc, const char * argv[])
{
	int opt;
	int mode = -1;

	while ((opt = getopt(argc, (char *const *)argv, "hvm:tl:")) != -1) {
		switch (opt) {
			case 'h':
				print_help(argv);
				break;
			case 'v':
				verbose = 1;
				break;
			case 'm':
				mode = atoi(optarg);
				break;
			case 't':
				inform_timestamps = 1;
				break;
			case 'l':
				max_images = atoi(optarg);
				break;
			default:
				print_help(argv);
				exit(EXIT_FAILURE);
		}
	}

	switch(mode) {
		case 1:
			with_cuda = 1;
			break;
		case 2:
			with_cpu = 1;
			break;
		case 3:
			with_cuda = 1;
			with_cpu = 1;
			break;
		default:
			fprintf(stderr, "No se especifico modo!\n");
			print_help(argv);
			exit(EXIT_FAILURE);
	}

	clearScreen();

	time_t startTime = time(NULL);
	if(verbose) {
		printf("Red neuronal de 3 capas para el reconocimiento de digitos numerico\n");
		printf("Configuracion: \n");
		printf("\tCUDA: %s\n", with_cuda == 1 ? "ON" : "OFF");
		printf("\tCPU: %s\n", with_cpu == 1 ? "ON" : "OFF");
		printf("\tRecord timestamps: %s\n", inform_timestamps == 1 ? "ON" : "OFF");
		printf("\tLimite: %d\n", max_images);
	}

	queue = utils_queue_alloc();

	input_data_u8 = (uint8_t *)calloc(1, 60000 * (28 * 28 + 1));
	input_data = (double *)calloc(1, sizeof(double) * 60000 * (28 * 28 + 1));
	input_labels = (uint8_t *)calloc(1, 60000);

	//Inicio thread de lectura
	uint64_t ts = get_time_usec();
	read_data();
	//printf("LECTURA DE DATOS:: %llu\n", get_time_usec() - ts);

	//Creo la red
	Network *nn = createNetwork(MNIST_IMG_HEIGHT * MNIST_IMG_WIDTH, 20, 10);
	Cuda_Network *cu_nn = cuda_create_network(MNIST_IMG_HEIGHT * MNIST_IMG_WIDTH, 20, 10);

	uint64_t ts1 = get_time_usec();
	cuda_copy_to_super_input(cu_nn, input_data);
	//printf("COPIA A SUPER INPUT:: %llu\n", get_time_usec() - ts1);

	trainNetwork(nn, cu_nn);

	testNetwork(nn, cu_nn);

	// Calculate and print the program's total execution time
	time_t endTime = time(NULL);
	double executionTime = difftime(endTime, startTime);
	if(inform_timestamps) {
		locateCursor(19, 1);
		printf("DONE! Tiempo total de ejecucion: %.1f seg\n",executionTime);
	}

	return 0;
}


