#include<stdio.h>
#include<stdint.h>
#include<stdlib.h>

#include <pmmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>


void convert_simd_avx(const int32_t * u, double * y, size_t n, double slope)
{
	const int32_t * u_end = u + n;
	const int32_t * u_current = u;
	double * y_current = y;
	
	__m128i mmx_u1, mmx_u2;
    __m256d mmx_y1, mmx_y2, mmx_y3, mmx_y4;
    __m256d mmx_slope_4 = _mm256_set1_pd(slope);
	
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
	
}