#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include <string.h>
#include <pmmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>

#define FILTER_ORDER 8
#define FRAME_LEN	2


#if defined(__i386__)
static __inline__ unsigned long long rdtsc(void)
{
    unsigned long long int x;
    __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
    return x;
}

#elif defined(__x86_64__)
static __inline__ unsigned long long rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}
#endif

void fir_simd(float* input, const float* coeff, float* output, size_t filter_order, unsigned long long* duration)
{

	int i=0, j=0, k=0;
	__m128 zero = _mm_setzero_ps();
	__m128 x0 = zero;
	__m128 x1 = zero;
	__m128 k0 = zero;
	__m128 k1 = zero;
	__m128 t0 = zero;
	__m128 t1 = zero;
	__m128 s0 = zero;
	__m128 s1 = zero;
	
	unsigned long long duration_local; 	
	unsigned long long duration_total = 0; 	
  
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
	for(j=0; j<filter_order; j++)
	{
		/* Generate input sample on 0th index */
		input[0] = j;
		
		duration_local = rdtsc();
		
		/********** Critical Path Begins **********/
		/* Unroll the loop */
		for(i=0; i<filter_order/8; i++)
		{
			/*~~~~~~~No Data Dependencies~~~~~~~~~~*/
			 k0 = _mm_load_ps(coeff + (i*8));
			 s0 = _mm_load_ps(input + (i*8));
			 k1 = _mm_load_ps(coeff + ( (i*8)+4) );
			 s1 = _mm_load_ps(input + ( (i*8)+4) );
			/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

			/*~~~~~~~~~~~~~~~~~~~~~*/
			 t0 = _mm_mul_ps(k0, s0);
			 t1 = _mm_mul_ps(k1, s1);
			/*~~~~~~~~~~~~~~~~~~~~~*/
			
			/*~~~~~~~~~~~~~~~~~~~~~*/
			 x1 = _mm_add_ps(x0, t0);
			/*~~~~~~~~~~~~~~~~~~~~~*/
			
			/*~~~~~~~~~~~~~~~~~~~~~*/
			 x0 = _mm_add_ps(x1, t1);
			/*~~~~~~~~~~~~~~~~~~~~~*/
		}
			
		/* horizontal add */
		x0 = _mm_hadd_ps(x0, zero);
		x0 = _mm_hadd_ps(x0, zero);
		
		/* Store the result */
		_mm_store_ss(output + j, x0);
		/********** Critical Path Ends **********/

		/* Get the cycle difference and add to total cycle count */
		duration_local = rdtsc() - duration_local;
		duration_total = duration_total + duration_local;

		/* Generate delay on input signal. i.e. shift all input elements to right */
		for(k=filter_order; k>0; k--)
				input[k] = input[k-1];		
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
	
	/* Get the average cycle count and store it in argument variable, duration */
	*duration = duration_total/filter_order;
	
	for(k=0; k<filter_order; k++)		
		input[k] = 0;
		
}

int compare_arrays(const float * u1, const float * u2, size_t n)
{
    size_t i = 0;
    while (i < n)
    {
        if (u1[i] != u2[i])
            return 0;
        ++i;
    }
    return 1;
}

int check_alignment(const void* p, int alignment, const char* name)
{
    size_t p_numeric = (size_t)p;
	
	/* If not aligned, return false */
    if (p_numeric % alignment != 0)
    	return 0;		           
	
	/* Aligned - return true */
	return 1;
}


float** prepare_coeff_matrix(const float* coeff, size_t n)
{

	float** ptr_array = malloc( FILTER_ORDER * sizeof(float));
	
	int idx=0, i=0;
	int isAligned = 0;
	float val = 0.0;
	
	ptr_array[0] = malloc(FILTER_ORDER * sizeof(float) + 16);
		
	/* check for 32-bit alignment. 32-bit alignment is required for SIMD AVX */
	isAligned = check_alignment(ptr_array, 32, "ptr_array");
	
	/* Align to 32-bit if not aligned */
	if(!isAligned)
		ptr_array[0] = (float *)((char*)ptr_array+16);
		
	for(i=0; i<n; i++)
		ptr_array[0][i] = coeff[i];		
	
	
	for(idx=1; idx<n-1; idx++)
	{
		ptr_array[idx] = malloc(FILTER_ORDER * sizeof(float) + 16);
		
		/* check for 32-bit alignment. 32-bit alignment is required for SIMD AVX */
		isAligned = check_alignment(ptr_array, 32, "ptr_array");
	
		/* Align to 32-bit if not aligned */
		if(!isAligned)
			ptr_array[idx] = (float *)((char*)ptr_array+16);
		
		val = 12;//ptr_array[idx-1][0];
		
		ptr_array[idx][n] = val;				
		
		for(i=0; i<n-1; i++)			
			ptr_array[idx][i] = ptr_array[idx-1][i+1];
		
		
	}
	
	return ptr_array;		
}

int main()
{
	
    unsigned long long duration_simd, duration_scalar;
	int isAligned = 0;
	int i=0, j=0;
	
	/* Coefficients array */
	float* coeff = malloc(FILTER_ORDER * sizeof(float));	
	
	/* Initialize some elments of coefficient array */
	coeff[0] = 2;
	coeff[3] = 3;
	coeff[FILTER_ORDER - 1] = 6;
	
	for(i=0; i<FILTER_ORDER; i++)
		coeff[i] = i+1;
	
	/* Prepare the coefficient 2-D matrix. Array of pointers are aligned to 32-bit by this function */ 
	float** coeff_mat = prepare_coeff_matrix(coeff, FILTER_ORDER);
	
	
	for(i=0; i<FILTER_ORDER; i++)
	{
		for(j=0; j<FILTER_ORDER; j++)
			printf("%f ", coeff_mat[i][j]);
		
		printf("\n");
	}
		
	
	
	/* check for 32-bit alignment. 32-bit alignment is required for SIMD AVX */
    isAligned = check_alignment(coeff, 32, "coeff");
	
	/* Align to 32-bit if not aligned */
	if(!isAligned)
		coeff = (float *)((char*)coeff+16);

	/* Input array */
    float* input = malloc(FILTER_ORDER * sizeof(float) + 16);	
	
	/* check for 32-bit alignment. 32-bit alignment is required for SIMD AVX */
    isAligned = check_alignment(input, 32, "input");
	
	/* Align to 32-bit if not aligned */
	if(!isAligned)
		input = (float *)((char*)input+16);

	/* Output array for vector operation*/
	float* output = malloc(FILTER_ORDER * sizeof(float) + 16);	
	
	/* check for 32-bit alignment */
    isAligned = check_alignment(output, 32, "output");
	
	/* Align to 32-bit if not aligned */
	if(!isAligned)
		output = (float *)((char*)output+16);
	
	/* Reference output array for scalar operation*/
	float* ref_out = malloc(FILTER_ORDER * sizeof(float));		
	
	/********* FIR operation in scalar *********/
	fir_scalar(input, coeff, ref_out, FILTER_ORDER,  &duration_scalar);
	
	/********* FIR operation in parallel *********/
	fir_simd(input, coeff, output, FILTER_ORDER,  &duration_simd);
	
	/* Sanity check */
    printf("\n Scalar Speed: %f.\n SIMD Speed:   %f.\n SIMD speedup: x%f.\n ", (double)duration_scalar / 1000, (double)duration_simd / 1000, (double)duration_scalar / (double)duration_simd);
    if (compare_arrays(output, ref_out, FILTER_ORDER) == 1)
        printf("Numerics OK\n\n");
    else
        printf("Numerics NOT OK\n\n");
	
	return 0;
}
