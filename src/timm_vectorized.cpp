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
#if 1
	printf("Using vectorized version with vector size = %d\n", n_floats);
#endif
	return;
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

float TimmVectorized::kernel(int cx, int cy, float* gradients, int ngradients)
{
	switch (instrSet)
	{
#ifdef SSE41_ENABLED
	case InstrSetSSE41:
		return kernelSSE41(cx, cy, gradients, ngradients);
#endif
#ifdef AVX_ENABLED
	case InstrSetAVX:
		return kernelAVX(cx, cy, gradients, ngradients);
#endif
#ifdef AVX2_ENABLED
	case InstrSetAVX2:
		return kernelAVX2(cx, cy, gradients, ngradients);
#endif
#ifdef AVX512_ENABLED
	case InstrSetAVX512F:
		return kernelAVX512(cx, cy, gradients, ngradients);
#endif
	default:
		return Timm::kernel(cx, cy, gradients, ngradients);
	}
}

