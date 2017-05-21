#pragma once

#include <pthread.h>
#include <stdlib.h>
#include "utils_queue.h"

struct Utils_Queue {
	struct Node *first, *last;
	int count;
	pthread_mutex_t mutex;
};

struct Node {
	void *ptr;
	struct Node *next;
};
