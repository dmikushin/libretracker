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

#ifdef __arm__
	inline float kernelOpNeon(float cx, float cy, const float* sd)
	{
		// is it faster with "static" ?
		static const float32x4_t zero = vdupq_n_f32(0.0f); 

		// set cx value into vector (vdupq_n_f32 == Load all lanes of vector to the same literal value)
		//float32x4_t cx_4 = vdupq_n_f32(cx);
		//float32x4_t cy_4 = vdupq_n_f32(cy);

		// loading memory into vector registers
		float32x4_t dx_in = vld1q_f32(sd);
		float32x4_t dy_in = vld1q_f32(sd +  4);
		float32x4_t gx_in = vld1q_f32(sd +  8);
		float32x4_t gy_in = vld1q_f32(sd + 12);

		// calc the difference vector
		//dx_in = vsubq_f32(dx_in, cx_4);
		//dy_in = vsubq_f32(dy_in, cy_4);
		dx_in = vmulq_n_f32(dx_in, cx);
		dy_in = vmulq_n_f32(dy_in, cy);

		// calc the dot product	
		float32x4_t tmp1 = vmulq_f32(dx_in, dx_in);
		float32x4_t tmp2 = vmulq_f32(dy_in, dy_in);
		tmp1 = vaddq_f32(tmp1, tmp2);

		// now cals the reciprocal square root
		tmp1 = vrsqrteq_f32(tmp1);

		// now normalize by multiplying
		dx_in = vmulq_f32(dx_in, tmp1);
		dy_in = vmulq_f32(dy_in, tmp1);

		// now calc the dot product with the gradient
		tmp1 = vmulq_f32(dx_in, gx_in);
		tmp2 = vmulq_f32(dy_in, gy_in);
		tmp1 = vaddq_f32(tmp1, tmp2);

		// now calc the maximum // does this really help ???
		tmp1 = vmaxq_f32(tmp1, zero);
		//tmp1 = vmaxq_n_f32(tmp1, 0.0f); // this instruction sadly does not exist
		
		// multiplication 
		tmp1 = vmulq_f32(tmp1, tmp1);

		//return vaddvq_f32(tmp1);
	
		// https://pmeerw.net/blog/programming/neon1.html
		// accumulate four quadword floats
		static const float32x2_t f0 = vdup_n_f32(0.0f);
		return vget_lane_f32(vpadd_f32(f0, vget_high_f32(tmp1) + vget_low_f32(tmp1)), 1);
	}
#endif // __arm__

#ifdef SSE41_ENABLED
	float kernelOpSSE41(float cx, float cy, const float* sd);
#endif
#ifdef AVX_ENABLED
	float kernelOpAVX(float cx, float cy, const float* sd);
#endif
#ifdef AVX2_ENABLED
	float kernelOpAVX2(float cx, float cy, const float* sd);
#endif
#ifdef AVX512_ENABLED
	float kernelOpAVX512(float cx, float cy, const float* sd);
#endif

protected :

	virtual unsigned int getSIMDVectorSize();

	virtual float* getGradients();

	virtual float kernel(float cx, float cy, float* gradients, int ngradients);

public :

	TimmVectorized();
};

#endif // TIMM_VECTORIZED_H

