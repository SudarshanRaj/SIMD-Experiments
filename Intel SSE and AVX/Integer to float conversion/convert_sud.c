#include<stdio.h>
#include<stdint.h>
#include<stdlib.h>
#include <string.h>

#if ( AVX || SSE)
#include <pmmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
#endif

#define RUN_COUNT	1000
#define _1KB            1024
#define _1MB            1048576
#define ARRAY_SIZE      (10*_1MB)

#define SLOPE           0.125

unsigned long long cycles1[RUN_COUNT], cycles2[RUN_COUNT];
unsigned long long total_cycles = 0;


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

//SIMD Implementation
//~~~~~~~~~~~~~~~~~ AVX ~~~~~~~~~~~~~~~~~~~~//
#if (AVX)
/* AVX implementation - 8 integer elements at a time */
void convert_simd(const int32_t * u, double * y, size_t n, unsigned long long * duration)
{
    unsigned long long duration_local; 	

    duration_local = rdtsc();

    const int32_t * u_end = u + n;
	const int32_t * u_current = u;
	double * y_current = y;

    __m128i mmx_u1, mmx_u2;
    __m256d mmx_y1, mmx_y2, mmx_y3, mmx_y4;
    __m256d mmx_slope_4 = _mm256_set1_pd(SLOPE);
	
	{
		for (; u_current < u_end; u_current += 8, y_current += 8)
		{
			/* Load 8 input values into an SSE register */
			mmx_u1 = _mm_load_si128(  (const __m128i *) u_current);
			mmx_u2 = _mm_load_si128(  (const __m128i *)  u_current+4);
		
			mmx_y1 = _mm256_cvtepi32_pd(mmx_u1);
			mmx_y2 = _mm256_cvtepi32_pd(mmx_u2);
			
			mmx_y3 = _mm256_mul_pd(mmx_y1, mmx_slope_4);    /* Apply slope */
			mmx_y4 = _mm256_mul_pd(mmx_y2, mmx_slope_4);    /* Apply slope */
			
			_mm256_store_pd(y_current, mmx_y3);
			_mm256_store_pd(y_current+4, mmx_y4);			
		}
	}    
    duration_local = rdtsc() - duration_local;
    *duration = duration_local;
	
}


//~~~~~~~~~~~~~~~~~ AVX ~~~~~~~~~~~~~~~~~~~~//

#elif (SSE)
//~~~~~~~~~~~~~~~~~ SSE3 ~~~~~~~~~~~~~~~~~~~~//
/* SSE3 implementation - 4 integer elements at a time */
void convert_simd(const int32_t * u, double * y, size_t n, unsigned long long * duration)
{
    unsigned long long duration_local; 	

    duration_local = rdtsc();

    const int32_t * u_end = u + n;
	const int32_t * u_current = u;
	double * y_current = y;

    __m128i mmx_u;
    __m128d mmx_y1, mmx_y2;
    __m128d mmx_slope_2 = _mm_set1_pd(SLOPE);

    for (; u_current < u_end; u_current += 4, y_current += 4)
    {
        /* Load 4 input values into an SSE register */
        mmx_u = _mm_load_si128((__m128i const*)u_current);

        /* Container conversion and scaling application */
        mmx_y1 = _mm_cvtepi32_pd(mmx_u);             /* Convert lower 2 integers to 2 doubles */
        mmx_y2 = _mm_mul_pd(mmx_y1, mmx_slope_2);    /* Apply slope */
        _mm_store_pd(y_current, mmx_y2);

        mmx_u  = _mm_srli_si128(mmx_u, 8);           /* Move integers 2 and 3 to positions 0 and 1 in their register */
        mmx_y1 = _mm_cvtepi32_pd(mmx_u);             /* Convert lower 2 integers to 2 doubles */
        mmx_y2 = _mm_mul_pd(mmx_y1, mmx_slope_2);    /* Apply slope */
        _mm_store_pd(y_current+2, mmx_y2);
    }

    duration_local = rdtsc() - duration_local;
    *duration = duration_local;
}

#endif

//~~~~~~~~~~~~~~~~~ SSE3 ~~~~~~~~~~~~~~~~~~~~//



void convert_scalar(const int32_t * u, double * y, size_t n, unsigned long long * duration)
{
	extern void convert_scalar_naive(const int32_t * u, double * y, size_t n, double slope);
    unsigned long long duration_local; 
	size_t i, j;
	
    duration_local = rdtsc();
    
    convert_scalar_naive(u, y, n, SLOPE);

    duration_local = rdtsc() - duration_local;
    *duration = duration_local;
}

int compare_arrays(const double * u1, const double * u2, size_t n)
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
	
	// If not aligned, return false
    if (p_numeric % alignment != 0)
    {
        //printf("%s not aligned to %d\n", name, alignment);
		return 0;		       
    }
	
	// Aligned - return true
	return 1;
}

int main()
{
	
    unsigned long long duration_simd, duration_scalar;
	int isAligned = 0;
	char* ptr = NULL;
	
    /* Input array */
    const int32_t * u = malloc(ARRAY_SIZE * sizeof(int32_t) + 16);
	//memset(u, 3, ARRAY_SIZE * sizeof(int32_t) + 16); //initialize u to all 1's
    isAligned = check_alignment(u, 32, "u");
	
	if(!isAligned)
		u = (int32_t *)((char*)u+16);
	isAligned = check_alignment(u, 32, "u");
	if(!isAligned)
	{
		printf("Alignment not happening. :( I am exiting...BYE!\n");
		exit(1);
	}			

    /* Output array for parallel conversion*/
    double * y = malloc(ARRAY_SIZE * sizeof(double) + 16);
	memset(y, 1, ARRAY_SIZE * sizeof(int32_t) + 16); //initialize y to all 1's
    isAligned = check_alignment(y, 32, "y");
	
	if(!isAligned)
		y = (double *)((char*)y+16);
	isAligned = check_alignment(y, 32, "y");
	if(!isAligned)
	{
		printf("Alignment not happening. :( I am exiting...BYE!\n");
		exit(1);
	}

    /* Reference array for serial conversio */
    double * r = malloc(ARRAY_SIZE * sizeof(double) + 16);
	memset(r, 2, ARRAY_SIZE * sizeof(int32_t) + 16); //initialize r to all 2's
    //check_alignment(r, 32, "r");

    /* Test conversion */
    convert_simd(u, y, ARRAY_SIZE, &duration_simd);

    /* Reference conversion */
    convert_scalar(u, r, ARRAY_SIZE, &duration_scalar);

    /* Sanity check */
    printf("Scalar speed: %f.\n  SIMD Speed: %f.\n SIMD speedup: x%f.\n ", (double)duration_scalar / 1000, (double)duration_simd / 1000, (double)duration_scalar / (double)duration_simd);
    if (compare_arrays(y, r, ARRAY_SIZE) == 1)
        printf("Numerics OK\n");
    else
        printf("Numerics NOT OK\n");
        
    
return 0;                    
	
}

