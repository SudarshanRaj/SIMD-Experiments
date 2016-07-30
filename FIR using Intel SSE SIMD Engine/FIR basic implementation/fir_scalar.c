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



void fir_scalar(float* input, const float* coef, float* output, size_t filter_order, unsigned long long* duration)
{
	unsigned long long duration_local;
	unsigned long long duration_total = 0;
	
	uint32_t i=0, j=0, k=0;
	float out_local = 0.0;	
   
   //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
	for(j=0; j<FRAME_LEN; j++)
	{
		/* Generate one input sample on 0th index */
		input[0] = j;
		
		duration_local = rdtsc_read();
		
		/********** Critical Path Begins **********/
		/* Filter operation */
		for(i=0; i<filter_order; i++)
		   out_local = out_local + (input[i]  * coef[i]);
	   /********** Critical Path Ends **********/
		/* Get the time cycle difference and add to total cycle count */
		duration_local = rdtsc_read() - duration_local;
		duration_total = duration_total + duration_local;		
	
		/* Store the output */
		output[j] = out_local;	
	
		/* Generate delay on input signal. i.e. shift all input elements to right */
		for(k=filter_order; k>0; k--)
				input[k] = input[k-1];
		
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
	
	/* Get the average cycle count and store it in argument variable, duration */
	*duration = duration_total/FRAME_LEN;
	
	
}