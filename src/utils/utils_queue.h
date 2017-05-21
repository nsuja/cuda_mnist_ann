#pragma once

typedef struct Utils_Queue Utils_Queue;

/**
 * Crea la cola
 */
Utils_Queue *utils_queue_alloc(void);

/**
 * Trae un paquete de la cola
 */
void *utils_queue_get(Utils_Queue *this);

/**
 * Inserta un paquete en cola
 */
int utils_queue_insert(Utils_Queue *this, void *ptr);

/**
 * Checkea si hay paquetes en cola
 * @return: -1: Error, 0: No hay, 1:Hay
 */
int utils_queue_is_empty(Utils_Queue *this);

/**
 * Devuelve cantidad de paquetes en cola
 * @return: -1: Error, >=0:Cantidad de paquetes
 */
int utils_queue_get_count(Utils_Queue *this);
