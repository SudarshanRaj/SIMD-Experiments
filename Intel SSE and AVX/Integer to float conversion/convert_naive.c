#include<stdio.h>
#include<stdint.h>
#include<stdlib.h>

void convert_scalar_naive(const int32_t * u, double * y, size_t n, double slope)
{
	size_t i, j;
	   
    for (i = 0; i < n; ++i)
    {
        *(y+i) = ((double)*(u+i) * slope);
    }
}