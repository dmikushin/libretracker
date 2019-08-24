#ifndef TIMM_MULTITHREADED_H
#define TIMM_MULTITHREADED_H

#include "timm_vectorized.h"

class TimmMultithreaded : public TimmVectorized
{
	int nthreads;

protected:

	virtual void pupilCenterKernel(cv::Mat& out_sum, float* gradients, int ngradients);

public :

	TimmMultithreaded();
};

#endif // TIMM_MULTITHREADED_H

