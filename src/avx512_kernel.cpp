#include "timm.h"

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#ifdef AVX512_ENABLED
float Timm::kernel_op_avx512(float cx, float cy, const float* sd)
{

	//__declspec(align(16)) float dx[4]; // no effect - compiler seems to automatically align code		

	__m512 zero = _mm512_set_ps(
		0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f); // this should be faster

	__m512 cx_sse = _mm512_set_ps(cx, cx, cx, cx, cx, cx, cx, cx, cx, cx, cx, cx, cx, cx, cx, cx);
	__m512 cy_sse = _mm512_set_ps(cy, cy, cy, cy, cy, cy, cy, cy, cy, cy, cy, cy, cy, cy, cy, cy);

	__m512 dx_in = _mm512_load_ps(sd);
	__m512 dy_in = _mm512_load_ps(sd+16);
	__m512 gx_in = _mm512_load_ps(sd+32);
	__m512 gy_in = _mm512_load_ps(sd+48);

	// calc the difference vector
	dx_in = _mm512_sub_ps(dx_in, cx_sse);
	dy_in = _mm512_sub_ps(dy_in, cy_sse);

	// calc the dot product	for the sixteen vec2f
	// Emits the Streaming SIMD Extensions 4 (SSE4) instruction dpps.
	// This instruction computes the dot product of single precision floating point values.
	// https://msdn.microsoft.com/en-ulibrary/bb514054(v=vs.120).aspx
	__m512 tmp1 = _mm512_mul_ps(dx_in, dx_in);
	__m512 tmp2 = _mm512_mul_ps(dy_in, dy_in);
	tmp1 = _mm512_add_ps(tmp1, tmp2);

	// now cals the reciprocal square root
	tmp1 = _mm512_rsqrt14_ps(tmp1);

	// now normalize by multiplying
	dx_in = _mm512_mul_ps(dx_in, tmp1);
	dy_in = _mm512_mul_ps(dy_in, tmp1);

	// now calc the dot product with the gradient
	tmp1 = _mm512_mul_ps(dx_in, gx_in);
	tmp2 = _mm512_mul_ps(dy_in, gy_in);
	tmp1 = _mm512_add_ps(tmp1, tmp2);

	// now calc the maximum // does this really help ???
	tmp1 = _mm512_max_ps(tmp1, zero);

	// multiplication 
	tmp1 = _mm512_mul_ps(tmp1, tmp1);

	// summation of all 16 floats
	return _mm512_reduce_add_ps(tmp1);

}
#endif // AVX512_ENABLED

