#include "utils_queue_priv.h"

Utils_Queue *utils_queue_alloc(void)
{
	Utils_Queue *q;
	q = calloc(1, sizeof(Utils_Queue));
	if(!q) {
		return NULL;
	}
	pthread_mutex_init(&q->mutex, NULL);
	return q;
}

int utils_queue_insert(Utils_Queue *this, void *ptr)
{
	struct Node *node = NULL;

	if(!this)
		return -1;

	node= malloc(sizeof(struct Node));
	if(!node) {
		return -1;
	}

	pthread_mutex_lock(&this->mutex);
	node->ptr = ptr;
	node->next = NULL;
	if(!this->first) {
		this->first = node;
		this->last = node;
	}
	else {
		this->last->next = node;
		this->last = node;
	}
	this->count++;
	pthread_mutex_unlock(&this->mutex);

	return 0;
}

void *utils_queue_get(Utils_Queue *this)
{
	struct Node *node;
	void *ret = NULL;

	if(!this || !this->first) {
		return NULL;
	}

	pthread_mutex_lock(&this->mutex);
	node = this->first;
	if(!node) {
		pthread_mutex_unlock(&this->mutex);
		return NULL;
	}
	ret = node->ptr;

	this->first = this->first->next;
	this->count--;

	free(node);
	pthread_mutex_unlock(&this->mutex);

	return ret;
}

int utils_queue_is_empty(Utils_Queue *this)
{
	if(!this)
		return -1;
	return this->first ? 0 : 1;
}

int utils_queue_get_count(Utils_Queue *this)
{
	if(!this)
		return -1;
	return this->count;
}
