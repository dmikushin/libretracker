#ifndef TIMM_GPU_H
#define TIMM_GPU_H

#include "timm_multithreaded.h"

class TimmGPU : public TimmMultithreaded
{
	bool gpuIsAvailable;

protected:

	virtual void pupilCenterKernel(cv::Mat& out_sum, float* gradients, int ngradients);

public :

	TimmGPU();
};

#endif // TIMM_GPU_H

