#ifdef ENABLE_GPU

#include "timm_gpu.h"
#include "GPU.h"

using namespace std;

TimmGPU::TimmGPU() : Timm(), failsafeTriggered(false),

ngradients(0), nout_sum(0), gradients(NULL), out_sum(NULL)

{
	if (GPU::isAvailable())
		gpuIsAvailable = true;
}

cv::Point TimmGPU::pupilCenter(const cv::Mat& eye_img)
{
	cv::Point result;
	if (!failsafeTriggered)
		result = Timm::pupilCenter(eye_img);
	
	// Not to be merged with the above, as the
	// failsafeTriggered flag may change.
	if (failsafeTriggered)
		return timmMultithreaded->pupilCenter(eye_img);
	
	return result;
}

void TimmGPU::pupilCenterKernel(cv::Mat& out_sum_, float* gradients_, int ngradients_)
{
	int nout_sum_ = out_sum_.cols * out_sum_.rows;

	if (gpuIsAvailable)
	{
		GPUerror_t status = { cudaSuccess };

		if ((ngradients < ngradients_) || (nout_sum < nout_sum_))
		{
			// Re-allocate everything.
			GPU::mfree();
			
			ngradients = ngradients_;
			nout_sum = nout_sum_;
			
			gradients = reinterpret_cast<float*>(GPU::malloc(ngradients * sizeof(float)));
			if (!gradients) goto failsafe;
			out_sum = reinterpret_cast<float*>(GPU::malloc(nout_sum * sizeof(float)));
			if (!out_sum) goto failsafe;
		}

		// Copy new gradients.
		status = GPU::memcpy(gradients, gradients_, ngradients * sizeof(float), memcpyHostToDevice);
		if ((status.cudaError != cudaSuccess) || (status.clError != CL_SUCCESS))
			goto failsafe;
		
		// Zero output array.
		status = GPU::memset(out_sum, 0, nout_sum * sizeof(float));
		if ((status.cudaError != cudaSuccess) || (status.clError != CL_SUCCESS))
			goto failsafe;
		
		// Launch GPU kernel.
		const unsigned int szblock = 128;
		dim3 nblocks = { nout_sum / szblock, 1, 1 };
		if (nblocks.x % szblock) nblocks.x++;
		status = GPU::launch(nblocks, szblock, 0, NULL,
			"kernelOpGPU", out_sum_.cols, out_sum, nout_sum, gradients, ngradients);
		if ((status.cudaError != cudaSuccess) || (status.clError != CL_SUCCESS))
			goto failsafe;
		
		status = GPU::synchronize();
		if ((status.cudaError != cudaSuccess) || (status.clError != CL_SUCCESS))
			goto failsafe;
		
		// Copy back the output array.
		status = GPU::memcpy(reinterpret_cast<float*>(out_sum_.data), out_sum,
			nout_sum * sizeof(float), memcpyDeviceToHost);
		if ((status.cudaError != cudaSuccess) || (status.clError != CL_SUCCESS))
			goto failsafe;
		
		return;
	}

failsafe :

	if (!failsafeTriggered)
	{
		fprintf(stderr, "Failsafe mode triggered in TimmGPU::pupilCenterKernel\n");
		failsafeTriggered = true;
		timmMultithreaded.reset(new TimmMultithreaded());
	}
}

#endif // ENABLE_GPU

