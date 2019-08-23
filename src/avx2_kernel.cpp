#include "timm.h"

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#ifdef AVX2_ENABLED
// https://stackoverflow.com/a/13222410/4063520
// x = ( x7, x6, x5, x4, x3, x2, x1, x0 )
inline float sum8(__m256 x)
{
	// hiQuad = ( x7, x6, x5, x4 )
	const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
	// loQuad = ( x3, x2, x1, x0 )
	const __m128 loQuad = _mm256_castps256_ps128(x);
	// sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
	const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
	// loDual = ( -, -, x1 + x5, x0 + x4 )
	const __m128 loDual = sumQuad;
	// hiDual = ( -, -, x3 + x7, x2 + x6 )
	const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
	// sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
	const __m128 sumDual = _mm_add_ps(loDual, hiDual);
	// lo = ( -, -, -, x0 + x2 + x4 + x6 )
	const __m128 lo = sumDual;
	// hi = ( -, -, -, x1 + x3 + x5 + x7 )
	const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
	// sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
	const __m128 sum = _mm_add_ss(lo, hi);
	return _mm_cvtss_f32(sum);
}

inline float sum8_alt(__m256 x)
{
	__m256 x2 = _mm256_permute2f128_ps(x, x, 1);
	x = _mm256_add_ps(x, x2);
	x = _mm256_hadd_ps(x, x);
	x = _mm256_hadd_ps(x, x);
	return _mm256_cvtss_f32(x);
}

float Timm::kernel_op_avx2(float cx, float cy, const float* sd)
{
	//__declspec(align(16)) float dx[4]; // no effect - compiler seems to automatically align code		

	__m256 zero = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f); // this should be faster

	__m256 cx_sse = _mm256_set_ps(cx, cx, cx, cx, cx, cx, cx, cx);
	__m256 cy_sse = _mm256_set_ps(cy, cy, cy, cy, cy, cy, cy, cy);

	__m256 dx_in = _mm256_load_ps(sd);
	__m256 dy_in = _mm256_load_ps(sd+8);
	__m256 gx_in = _mm256_load_ps(sd+16);
	__m256 gy_in = _mm256_load_ps(sd+24);

	// calc the difference vector
	dx_in = _mm256_sub_ps(dx_in, cx_sse);
	dy_in = _mm256_sub_ps(dy_in, cy_sse);

	// calc the dot product	for the eight vec2f
	// Emits the Streaming SIMD Extensions 4 (SSE4) instruction dpps.
	// This instruction computes the dot product of single precision floating point values.
	// https://msdn.microsoft.com/en-ulibrary/bb514054(v=vs.120).aspx
	__m256 tmp1 = _mm256_mul_ps(dx_in, dx_in);
	__m256 tmp2 = _mm256_mul_ps(dy_in, dy_in);
	tmp1 = _mm256_add_ps(tmp1, tmp2);

	// now cals the reciprocal square root
	tmp1 = _mm256_rsqrt_ps(tmp1);

	// now normalize by multiplying
	dx_in = _mm256_mul_ps(dx_in, tmp1);
	dy_in = _mm256_mul_ps(dy_in, tmp1);

	// now calc the dot product with the gradient
	tmp1 = _mm256_mul_ps(dx_in, gx_in);
	tmp2 = _mm256_mul_ps(dy_in, gy_in);
	tmp1 = _mm256_add_ps(tmp1, tmp2);

	// now calc the maximum // does this really help ???
	tmp1 = _mm256_max_ps(tmp1, zero);

	// multiplication 
	tmp1 = _mm256_mul_ps(tmp1, tmp1);

	return sum8(tmp1); // a tiny bit faster
	//return sum8_alt(tmp1);
}
#endif // AVX2_ENABLED

