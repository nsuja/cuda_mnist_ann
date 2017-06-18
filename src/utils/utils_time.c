#include <stddef.h>
#include <stdint.h>
#include <sys/time.h>

uint64_t get_time_usec()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000000ULL + tv.tv_usec;
}
