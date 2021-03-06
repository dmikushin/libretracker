#include "timm_vectorized.h"

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#ifdef SSE41_ENABLED
float TimmVectorized::kernelSSE41(int cx, int cy, float* gradients, int ngradients)
{
	// wenn das gut klappt, dann für raspi mal die sse2neon lib anschauen: https://github.com/jratcliff63367/sse2neon

	__m128 zero = _mm_set_ps(0.0f, 0.0f, 0.0f, 0.0f); // this should be faster

	__m128 cx_sse = _mm_set_ps(cx, cx, cx, cx);
	__m128 cy_sse = _mm_set_ps(cy, cy, cy, cy);

	__m128 c_out = zero;

	for (int i = 0; i < ngradients; i += 4 * n_floats)
	{
		const float* sd = &gradients[i];

		__m128 dx_in = _mm_load_ps(sd);
		__m128 dy_in = _mm_load_ps(sd+4);
		__m128 gx_in = _mm_load_ps(sd+8);
		__m128 gy_in = _mm_load_ps(sd+12);

		// calc the dot product	for the four vec2f
		// Emits the Streaming SIMD Extensions 4 (SSE4) instruction dpps.
		// This instruction computes the dot product of single precision floating point values.
		// https://msdn.microsoft.com/en-ulibrary/bb514054(v=vs.120).aspx
		dx_in = _mm_sub_ps(dx_in, cx_sse);
		dy_in = _mm_sub_ps(dy_in, cy_sse);
		__m128 tmp1 = _mm_mul_ps(dx_in, dx_in);
		__m128 tmp2 = _mm_mul_ps(dy_in, dy_in);
		tmp1 = _mm_add_ps(tmp1, tmp2);
		
		// now cals the reciprocal square root
		tmp1 = _mm_rcp_ps(tmp1);
		
		// now calc the dot product with the gradient
		dx_in = _mm_mul_ps(dx_in, gx_in);
		dy_in = _mm_mul_ps(dy_in, gy_in);

		// now normalize by multiplying
		dx_in = _mm_mul_ps(dx_in, tmp1);
		dy_in = _mm_mul_ps(dy_in, tmp1);
		tmp1 = _mm_add_ps(dx_in, dy_in);

		// now calc the maximum // does this really help ???
		tmp1 = _mm_max_ps(tmp1, zero);

		// multiplication 
		tmp1 = _mm_mul_ps(tmp1, tmp1);

		c_out = _mm_add_ps(c_out, tmp1);
	}

	// and finally, summation of all 4 floats
	// two horizontal adds do the trick:)
	c_out = _mm_hadd_ps(c_out, c_out);
	c_out = _mm_hadd_ps(c_out, c_out);
	
	return _mm_cvtss_f32(c_out);
}
#endif // SSE41_ENABLED

