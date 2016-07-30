#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include <string.h>
#include <pmmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>

#define FILTER_ORDER 	10*1024
#define FRAME_LEN	128

#define CHECK_FOR_NULL(x, y)	do{ if(x==NULL) {printf("\nError!! Pointer %s is NULL. Exiting...\n", y); exit(1);} }while(0);

float coeff_mat[FILTER_ORDER][FILTER_ORDER] __attribute__((aligned(32)));

extern void fir_scalar(float* input, float* coeff_mat, float* output, size_t filter_order, unsigned long long* duration);

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

typedef struct system_pointers
{
	float* coeff;
	float* input_scalar;
	float* input_simd;
	float* output_scalar;
	float* output_simd;
}system_pointers;

void fir_simd(float* input, const float* coeff_mat, float* output, size_t filter_order, unsigned long long* duration)
{

	int i=0, j=0;
	__m128 zero = _mm_setzero_ps();
	__m128 x0 = zero;
	__m128 x1 = zero;
	__m128 k0 = zero;
	__m128 k1 = zero;
	__m128 t0 = zero;
	__m128 t1 = zero;
	__m128 s0 = zero;
	__m128 s1 = zero;

	/* Index pointer to insert new input sample into buffer */
	uint32_t input_buffer_idx = 0;
	
	/* Index pointer to coeff buffer */
	uint32_t coeff_buffer_idx = 0;
	
	unsigned long long duration_local; 	
	unsigned long long duration_total = 0; 	
	
	duration_local = rdtsc();
	
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
	for(j=0; j<FRAME_LEN; j++)
	{
		/* Generate input sample on head index */
		input[input_buffer_idx] = j;
		
		//duration_local = rdtsc();
		
		/********** Critical Path Begins **********/
		/* Unroll the loop */
		for(i=0; i<filter_order; i+=8)
		{
			/*~~~~~~~No Data Dependencies~~~~~~~~~~*/
			 k0 = _mm_load_ps(coeff_mat + coeff_buffer_idx);
			 s0 = _mm_load_ps(input+i);
			
			 coeff_buffer_idx = coeff_buffer_idx + 4;

			 k1 = _mm_load_ps(coeff_mat + coeff_buffer_idx);
			 s1 = _mm_load_ps(input + i+4);
			/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

			/*~~~~~~~~~~~~~~~~~~~~~*/
			 t0 = _mm_mul_ps(k0, s0);
			 t1 = _mm_mul_ps(k1, s1);
			/*~~~~~~~~~~~~~~~~~~~~~*/
			
			/*~~~~~~~~~~~~~~~~~~~~~*/
			 x1 = _mm_add_ps(x0, t0);
			/*~~~~~~~~~~~~~~~~~~~~~*/
			
			coeff_buffer_idx = coeff_buffer_idx + 4;
			
			/*~~~~~~~~~~~~~~~~~~~~~*/
			 x0 = _mm_add_ps(x1, t1);
			/*~~~~~~~~~~~~~~~~~~~~~*/
		}
			
		/* horizontal add */
		x0 = _mm_hadd_ps(x0, zero);

		/* ring buffer advance */
		if(input_buffer_idx == 0) 
			input_buffer_idx = filter_order - 1;
		else
			--input_buffer_idx;

		if(coeff_buffer_idx == (filter_order * filter_order) )
			coeff_buffer_idx = 0;

		x0 = _mm_hadd_ps(x0, zero);
		
		/* Store the result */
		_mm_store_ss(output + j, x0);
		/********** Critical Path Ends **********/

		/* Get the cycle difference and add to total cycle count */
		//duration_local = rdtsc() - duration_local;
		//duration_total = duration_total + duration_local;	
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
	
	/* Get the average cycle count and store it in argument variable, duration */
	//*duration = duration_total/FRAME_LEN;
	duration_local = rdtsc() - duration_local;
	*duration = duration_local;
	
}

int compare_arrays(const float * u1, const float * u2, size_t n)
{
    size_t i = 0;
    while (i < n)
    {
        if (u1[i] != u2[i])
	{
		printf("Numerics Error!!\n");
		printf(" 0x%016llx",     *(unsigned long long *)&(u1[i]));
		printf(" =! 0x%016llx", *(unsigned long long *)&(u2[i]));
		printf(" at index %d\n", (int)i);
		return 0;
	}            
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

void prepare_coeff_matrix(float* coeff_mat, const float* coeff, size_t n)
{
	int i=0, j=0;
	float val=0;
	
	for(i=0; i<n; i++)
		coeff_mat[i] = coeff[i];
		
	for(i=1; i<n; i++)
	{
		val = coeff_mat[(i-1)*n];
		
		for(j=0; j<n-1; j++)		
			coeff_mat[(i*n)+j] = coeff_mat[((i-1)*n)+j+1];
				
		coeff_mat[(i*n)+n-1] = val;
	}			
}

void free_system_pointers(system_pointers* sys_ptrs)
{
	free(sys_ptrs->coeff);
	free(sys_ptrs->input_scalar);
	free(sys_ptrs->input_simd);
	free(sys_ptrs->output_scalar);
	free(sys_ptrs->output_simd);
}

void free_coeff_matrix(float** coeff_mat, size_t n)
{
	int i=0;
	
	for(i=0; i<n; i++)
		free(coeff_mat[i]);
}

int main()
{
	
    unsigned long long duration_simd, duration_scalar;
	int isAligned = 0;
	int i=0, j=0;
	
	system_pointers sys_ptrs;
	
	/****** Allocate memory for input and output buffers ******/	
	/* Coefficients array */
	float*  coeff = malloc(FILTER_ORDER * sizeof(float));	
	CHECK_FOR_NULL(coeff, "coeff");
	
	/* Coefficients array */
	//float*  coeff_mat = malloc((unsigned long long)FILTER_ORDER * FILTER_ORDER * sizeof(float)  + 16);	
	//CHECK_FOR_NULL(coeff_mat, "coeff_mat");
	
	/* Initialize some elments of coefficient array */
	coeff[0] = 2;
	coeff[3] = 3;
	coeff[FILTER_ORDER - 1] = 6;
	
	/* Input array for Scalar FIR operation */
    float* input_scalar = malloc(FILTER_ORDER * sizeof(float) + 16);
	CHECK_FOR_NULL(input_scalar, "input_scalar");
	memset(input_scalar,0, FILTER_ORDER*sizeof(float) + 16);	
	
	/* Input array for Vector FIR operation */
    float* input_simd = malloc(FILTER_ORDER * sizeof(float) + 16);
	CHECK_FOR_NULL(input_simd, "input_simd");
	memset(input_simd,0, FILTER_ORDER*sizeof(float) + 16);
	
	/* Output array for vector operation*/
	float* output_simd = malloc(FRAME_LEN * sizeof(float) + 16);		
	CHECK_FOR_NULL(output_simd, "output_simd");
	memset(output_simd,1, FRAME_LEN*sizeof(float) + 16);
	
	/* Reference output array for scalar operation*/
	float* ref_out = malloc(FRAME_LEN * sizeof(float));	
	CHECK_FOR_NULL(ref_out, "ref_out");
	memset(ref_out,3, FRAME_LEN*sizeof(float));
	
	/* Store pointers for freeing in the future */
	sys_ptrs.coeff = coeff;
	sys_ptrs.input_scalar = input_scalar;
	sys_ptrs.input_simd = input_simd;
	sys_ptrs.output_simd = output_simd;		
	sys_ptrs.output_scalar = ref_out;		

	/****** Check for alignment for buffers used in SIMD ******/	
	/* check for 32-bit alignment. 32-bit alignment is required for SIMD AVX */
    //isAligned = check_alignment(coeff_mat, 32, "coeff_mat");
	
	/* Align to 32-bit if not aligned */
	//if(!isAligned)
		//coeff_mat = (float *)((char*)&coeff_mat[0]+16);	
	
	/* check for 32-bit alignment. 32-bit alignment is required for SIMD AVX */
    isAligned = check_alignment(input_simd, 32, "input_simd");
	
	/* Align to 32-bit if not aligned */
	if(!isAligned)
		input_simd = (float *)((char*)input_simd+16);

	/* check for 32-bit alignment */
    isAligned = check_alignment(output_simd, 32, "output");
	
	/* Align to 32-bit if not aligned */
	if(!isAligned)
		output_simd = (float *)((char*)output_simd+16);

	/* Prepare the coefficient 2-D matrix. Array of pointers returned are aligned to 32-bit by this function */ 
	prepare_coeff_matrix(&coeff_mat[0][0], coeff, FILTER_ORDER);	
	
	/********* FIR operation in parallel *********/
	fir_simd(input_simd, &coeff_mat[0][0], output_simd, FILTER_ORDER,  &duration_simd);	

	/********* FIR operation in scalar *********/
	fir_scalar(input_scalar, &coeff_mat[0][0], ref_out, FILTER_ORDER,  &duration_scalar);
	
	/* Sanity check */
    printf("\n Scalar Speed: %f.\n SIMD Speed:   %f.\n SIMD speedup: x%f.\n ", (double)duration_scalar / 1000, (double)duration_simd / 1000, (double)duration_scalar / (double)duration_simd);
    if (compare_arrays(output_simd, ref_out, FRAME_LEN) == 1)
        printf("Numerics OK\n\n");
    else
        printf(" Numerics NOT OK\n\n");
	
	/* Free all the pointers */
	free_system_pointers(&sys_ptrs);

	return 0;
}
