#ifndef TIMM_GPU_H
#define TIMM_GPU_H

#include "timm_multithreaded.h"

#include <memory>

class TimmGPU : public Timm
{
	// Fallback mode.
	bool failsafeTriggered;
	std::unique_ptr<TimmMultithreaded> timmMultithreaded;

	bool gpuIsAvailable;

	float* gradients;
	float* out_sum;
	
	int ngradients, nout_sum;

protected:

	virtual void pupilCenterKernel(cv::Mat& out_sum, float* gradients, int ngradients);

public :

	TimmGPU();

	// estimates the pupil center
	// inputs: eye image
	virtual cv::Point pupilCenter(const cv::Mat& eye_img);
};

#endif // TIMM_GPU_H

