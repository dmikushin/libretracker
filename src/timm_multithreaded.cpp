#include "timm_multithreaded.h"
#include "timing.h"

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>
#include <tbb/task_scheduler_init.h>

using namespace std;

TimmMultithreaded::TimmMultithreaded() : TimmVectorized()
{
	nthreads = 1;

	// Get threads from the conventional OMP_NUM_THREADS environment variable.
	char* c_omp_num_threads	= getenv("OMP_NUM_THREADS");
	if (c_omp_num_threads)
	{
		int omp_num_threads = atoi(c_omp_num_threads);
		if (omp_num_threads > 0)
			nthreads = omp_num_threads;
	}

	// TODO Must be alive during the whole application scope, in order to have effect!
	tbb::task_scheduler_init init(nthreads);
#if 0
	printf("# of threads : %d\n", nthreads);
#endif
}

void TimmMultithreaded::pupilCenterKernel(cv::Mat& out_sum, float* gradients, int ngradients)
{
	auto data = out_sum.ptr<float>(0);

	if (nthreads > 1)
	{
		tbb::parallel_for(tbb::blocked_range2d<unsigned int>(0, out_sum.rows, 0, out_sum.cols),
			[&](const tbb::blocked_range2d<unsigned int>& r)
		{
			for (unsigned int y = r.rows().begin(); y != r.rows().end(); y++)
				for (unsigned int x = r.cols().begin(); x != r.cols().end(); x++)
					data[y * out_sum.cols + x] = kernel(x, y, gradients, ngradients);
		});
	}
	else
	{
		for (unsigned int y = 0; y < out_sum.rows; y++)
			for (unsigned int x = 0; x < out_sum.cols; x++)
				data[y * out_sum.cols + x] = kernel(x, y, gradients, ngradients);
	}
}

