#include "DeviceCode.h"

#if defined(__CUDACC__)
extern "C"
#endif
__global__ void kernelOpGPU(int width, uint64_t out_sum_, int nout_sum, uint64_t gradients_, int ngradients)
{
	float* out_sum = (float*)out_sum_;
	float* gradients = (float*)gradients_;

	int idx = threadIdx(x) + blockDim(x) * blockIdx(x);
	if (idx >= nout_sum) return;
	
	int cy = idx / width;
	int cx = idx - cy * width;

	float c_out = 0.0f;
	
	for (size_t i = 0; i < ngradients; i += 4)
	{
		float* sd = &gradients[i];

		float x  = sd[0];
		float y  = sd[1];
		float gx = sd[2];
		float gy = sd[3];
		
		float dx = x - cx;
		float dy = y - cy;

		// normalize d
		float magnitude = (dx * dx) + (dy * dy);

		magnitude = 1.0f / sqrt(magnitude);

		dx = dx * magnitude;
		dy = dy * magnitude;

		float dotProduct = dx * gx + dy * gy;
		dotProduct = max(0.0f, dotProduct);

		c_out += dotProduct * dotProduct;
	}
	
	out_sum[cy * width + cx] = c_out;
}

