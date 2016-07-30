#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#define FRAME_LEN	128

#if defined(__i386__)
static __inline__ unsigned long long rdtsc_read(void)
{
    unsigned long long int x;
    __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
    return x;
}

#elif defined(__x86_64__)
static __inline__ unsigned long long rdtsc_read(void)
{
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}
#endif


void fir_scalar(float* input, const float* coeff_mat, float* output, size_t filter_order, unsigned long long* duration)
{
	unsigned long long duration_local;
	unsigned long long duration_total = 0;
	
	uint32_t i=0, j=0, k=0;
	float out_local = 0.0;	

	/* Index pointer to insert new input sample into buffer */
	uint32_t input_buffer_idx = 0;
	
	/* Index pointer to coeff buffer */
	uint32_t coeff_buffer_idx = 0;
     
	duration_local = rdtsc_read();
	 
   	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
	for(j=0; j<FRAME_LEN; j++)
	{
		/* Generate one input sample on 0th index */
		input[input_buffer_idx] = j;
		
		//duration_local = rdtsc_read();

		/********** Critical Path Begins **********/
		/* Filter operation */
		for(i=0; i<filter_order; i++, coeff_buffer_idx++)
		   out_local = out_local + (input[i]  * coeff_mat[coeff_buffer_idx]);	  
		/********** Critical Path Ends **********/

		/* Get the time cycle difference and add to total cycle count */
		//duration_local = rdtsc_read() - duration_local;
		//duration_total = duration_total + duration_local;		
	
		/* ring buffer advance */
		if(input_buffer_idx == 0) 
			input_buffer_idx = filter_order - 1;
		else
			--input_buffer_idx;

		if(coeff_buffer_idx == (filter_order * filter_order) )
			coeff_buffer_idx = 0;

		/* Store the output */
		output[j] = out_local;
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
	
	/* Get the average cycle count and store it in argument variable, duration */
	//*duration = duration_total/FRAME_LEN;	
	
	duration_local = rdtsc_read() - duration_local;
	*duration = duration_local;
	
}
