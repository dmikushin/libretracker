#ifndef TIMM_VECTORIZED_H
#define TIMM_VECTORIZED_H

// needed to access Vector Extension Instructions
#ifdef __arm__
#include <arm_neon.h>
#endif

#include "timm.h"
#include "instrset.h"
#include "aligned_allocator.h"

class TimmVectorized : public Timm
{
	int n_floats;

	// Specifies instruction set: AVX512, AVX2, AVX, SSE4.1, 80386.
	agner::InstrSet instrSet;

	// optimised for SIMD: 
	// this vector stores sequential chunks of floats for x, y, gx, gy
	std::vector<float, AlignedAllocator<float, Alignment::AVX512> > gradients;

#ifdef SSE41_ENABLED
	float kernelSSE41(float cx, float cy, float* gradients, int ngradients);
#endif
#ifdef AVX_ENABLED
	float kernelAVX(float cx, float cy, float* gradients, int ngradients);
#endif
#ifdef AVX2_ENABLED
	float kernelAVX2(float cx, float cy, float* gradients, int ngradients);
#endif
#ifdef AVX512_ENABLED
	float kernelAVX512(float cx, float cy, float* gradients, int ngradients);
#endif

protected :

	virtual unsigned int getSIMDVectorSize();

	virtual float* getGradients();

	virtual float kernel(float cx, float cy, float* gradients, int ngradients);

public :

	TimmVectorized();
};

#endif // TIMM_VECTORIZED_H

