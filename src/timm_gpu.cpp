#include "timm_gpu.h"
#include "GPU.h"

using namespace std;

TimmGPU::TimmGPU() : TimmMultithreaded()
{
	if (GPU::isAvailable())
		gpuIsAvailable = true;
}

void TimmGPU::pupilCenterKernel(cv::Mat& out_sum, float* gradients, int ngradients)
{
#if 0
	if (gpuIsAvailable)
	{
		auto data = out_sum.ptr<float>(0);
	}
	else
		TimmMultithreaded::pupilCenterKernel(out_sum, gradients, ngradients);
#else
	TimmMultithreaded::pupilCenterKernel(out_sum, gradients, ngradients);
#endif
}

