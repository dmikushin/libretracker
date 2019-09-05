#include "timm_vectorized.h"

using namespace agner;
using namespace std;

TimmVectorized::TimmVectorized() : Timm()
{
	instrSet = InstrSet80386;
	n_floats = 1;
#ifdef AVX512_ENABLED
	if (isSupported(InstrSetAVX512F))
	{
		instrSet = InstrSetAVX512F;
		n_floats = 16;
		goto finish;
	}
#endif
#ifdef AVX2_ENABLED
	if (isSupported(InstrSetAVX2))
	{
		instrSet = InstrSetAVX2;
		n_floats = 8;
		goto finish;
	}
#endif
#ifdef AVX_ENABLED
	if (isSupported(InstrSetAVX))
	{
		instrSet = InstrSetAVX;
		n_floats = 8;
		goto finish;
	}
#endif
	if (isSupported(InstrSetSSE41))
	{
		instrSet = InstrSetSSE41;
		n_floats = 4;
		goto finish;
	}
finish :
	printf("instrSet = %d\n", instrSet);
}

float* TimmVectorized::getGradients()
{
	if (!gradients.size())
		gradients.resize(scaledWidth() * scaledHeight() * n_floats * 4);

	return reinterpret_cast<float*>(&gradients[0]);
}

unsigned int TimmVectorized::getSIMDVectorSize()
{
	return n_floats;
}

float TimmVectorized::kernel(float cx, float cy, float* gradients, int ngradients)
{
	float c_out = 0.0f;

#ifdef __arm__
	for (size_t i = 0; i < ngradients; i += 4 * n_floats)
		c_out += kernelOp_arm128(cx, cy, &gradients[i]);
#else
	switch (instrSet)
	{
#ifdef SSE41_ENABLED
	case InstrSetSSE41:
		for (size_t i = 0; i < ngradients; i += 4 * n_floats)
			c_out += kernelOpSSE41(cx, cy, &gradients[i]);
		break;
#endif
#ifdef AVX_ENABLED
	case InstrSetAVX:
		for (size_t i = 0; i < ngradients; i += 4 * n_floats)
			c_out += kernelOpAVX(cx, cy, &gradients[i]);
		break;
#endif
#ifdef AVX2_ENABLED
	case InstrSetAVX2:
		for (size_t i = 0; i < ngradients; i += 4 * n_floats)
			c_out += kernelOpAVX2(cx, cy, &gradients[i]);
		break;
#endif
#ifdef AVX512_ENABLED
	case InstrSetAVX512F:
		for (size_t i = 0; i < ngradients; i += 4 * n_floats)
			c_out += kernelOpAVX512(cx, cy, &gradients[i]);
		break;
#endif
#endif
	default:
		c_out = Timm::kernel(cx, cy, gradients, ngradients);
		break;
	}
	return c_out;
}

