#include "timm.h"
#include "instrset.h"
#include "timing.h"

#include <Eigen/Eigen>
#include <queue>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>

using namespace agner;
using namespace std;

Timm::Timm()
{
#ifdef OPENCL_ENABLED
	simd_width = USE_OPENCL;
#else
	simd_width = USE_NO_VEC;
#ifdef AVX512_ENABLED
	if (isSupported(InstrSetAVX512F))
	{
		simd_width = USE_VEC512;
		goto finish;
	}
#endif
#ifdef AVX2_ENABLED
	if (isSupported(InstrSetAVX2))
	{
		simd_width = USE_VEC256;
		goto finish;
	}
#endif
	if (isSupported(InstrSetSSE41))
	{
		simd_width = USE_VEC128;
		goto finish;
	}
#endif
finish :
	printf("simd_width = %d\n", simd_width);
}

cv::Point Timm::pupil_center(const cv::Mat& eye_img)
{
	double t1; get_time(&t1);

	pre_process(eye_img);

	//*
	double t1b; get_time(&t1b);
	#ifdef _WIN32
	_ReadWriteBarrier();
	#endif
	prepare_data();

	// faster code using hand optimized objective function		
	// todo: parallelize with std::async threadpools
	//auto cols = outSum.cols;
	auto data = out_sum.ptr<float>(0);
	size_t idx = 0;

	// https://docs.microsoft.com/en-us/cpp/parallel/auto-parallelization-and-auto-vectorization
	// compiler switch must be enabled  /Qpar /Qpar-report:1 
	// #pragma loop(hint_parallel(2))

	const size_t cols = out_sum.cols;
	auto pixel_loop = [&](const int y1, const int y2)
	{
		//for (size_t y = 0; y < out_sum.rows; y++)
		for (size_t y = y1; y <= y2; y++)
		{
			for (size_t x = 0; x < out_sum.cols; x++)
			{
				//data[y*cols + x] = calc_objective_function(x, y, gradientX, gradientY); // 70.1 ms
				//data[y*cols + x] = calc_objective_function_cache_friendly(x, y, gradients); // 11.5 ms
				data[y*cols + x] = kernel(x, y, gradients);
			}
		}
	};

	if (n_threads > 1)
	{
		// create threads
		threads.clear();
		int block_size = ceil(float(out_sum.rows) / float(n_threads));
		for (int i = 0; i < n_threads; i++)
		{
			int y1 = i * block_size;
			int y2 = min((i + 1) * block_size - 1, out_sum.rows-1);
			threads.emplace_back( thread(pixel_loop, y1, y2) );
		}
		// wait for completion of all threads
		for (auto& t : threads) { t.join(); }
	}
	else
	{
		pixel_loop(0, out_sum.rows - 1);
	}

	cv::multiply(out_sum, weight_float, out);
	#ifdef _WIN32
	_ReadWriteBarrier(); // to avoid instruction reordering - important for accurate timings
	#endif
	double t2b; get_time(&t2b);
	measure_timings[1] = t2b - t1b;
	//*/


	/*
	//  original code Tristan Hume, 2012, except one change: here, cv:Mats are float instead of double (already saving considerable computation time)
	// kept in here for reference and for speed comparison.
	timer2.tick(); _ReadWriteBarrier();
	for (int y = 0; y < weight_float.rows; ++y) {
		const float *Xr = gradient_x.ptr<float>(y), *Yr = gradient_y.ptr<float>(y);
		for (int x = 0; x < weight_float.cols; ++x) {
			float gX = Xr[x], gY = Yr[x];
			if (gX == 0.0 && gY == 0.0) {
				continue;
			}
			testPossibleCentersFormula(x, y, weight, gX, gY, out_sum);
		}
	}
	out = out_sum;
	_ReadWriteBarrier(); measure_timings[1] = timer2.tock(false);
	//*/

	cv::Point result = undo_scaling(post_process(), eye_img.cols);

	double t2; get_time(&t2);
	measure_timings[0] = t2 - t1;
	
	return result;
}


float Timm::kernel(float cx, float cy, const vector<float>& gradients)
{
	float c_out = 0.0f;
	const size_t s = gradients.size();

	const size_t n_floats = simd_width / (8 * sizeof(float));

	switch (simd_width)
	{
#ifdef __arm__
	case USE_VEC128: for (size_t i = 0; i < s; i += 4 * n_floats) { c_out += kernel_op_arm128(cx, cy, &gradients[i]); } break;
#else
	case USE_NO_VEC: for (size_t i = 0; i < s; i += 4 * n_floats) { c_out += kernel_op(cx, cy, &gradients[i]); } break;
	case USE_VEC128: for (size_t i = 0; i < s; i += 4 * n_floats) { c_out += kernel_op_sse(cx, cy, &gradients[i]); } break;
	case USE_VEC256: for (size_t i = 0; i < s; i += 4 * n_floats) { c_out += kernel_op_avx2(cx, cy, &gradients[i]); } break;
	case USE_VEC512: for (size_t i = 0; i < s; i += 4 * n_floats) { c_out += kernel_op_avx512(cx, cy, &gradients[i]); } break;
#endif
	default: throw std::invalid_argument("wrong or unsupported vectorization width in Timm::kernel"); break;
	}
	return c_out;
}


float Timm::calc_dynamic_threshold(const cv::Mat &mat, float stdDevFactor)
{
	cv::Scalar stdMagnGrad, meanMagnGrad;
	cv::meanStdDev(mat, meanMagnGrad, stdMagnGrad);
	float stdDev = stdMagnGrad[0] / sqrt(mat.rows*mat.cols);
	return stdDevFactor * stdDev + meanMagnGrad[0];
}


void Timm::pre_process(const cv::Mat& img)
{
	#ifdef _WIN32
	_ReadWriteBarrier();
	#endif
	// down sample to speed up
	cv::resize(img, img_scaled, cv::Size(opt.down_scaling_width, img.rows * float(opt.down_scaling_width) / img.cols));


	// calc the gradients 0.06ms
	// valid inputs to sobel kernel size: -1, 1, 3, 5, 7 
	cv::Sobel(img_scaled, gradient_x, CV_32F, 1, 0, opt.sobel);
	cv::Sobel(img_scaled, gradient_y, CV_32F, 0, 1, opt.sobel);

	// compute all the magnitudes 0.01ms
	mags = gradient_x.mul(gradient_x) + gradient_y.mul(gradient_y);
	sqrt(mags, mags); // is sqrt really necessary ??



	//-- Normalize and threshold the gradient
	//compute the threshold 0.004ms
	float gradientThresh = calc_dynamic_threshold(mags, opt.gradient_threshold);
	//float gradientThresh = kGradientThreshold;
	//float gradientThresh = 0;



	// normalize 0.007ms
	gradient_x = gradient_x / mags;
	gradient_y = gradient_y / mags;



	// set all values smaller than the threshold to zero 0.02ms

	auto tmp = mags < gradientThresh;
	gradient_x.setTo(0.0f, tmp);
	gradient_y.setTo(0.0f, tmp);

	//-- Create a blurred and inverted image for weighting (see paper) 0.024ms
	if (opt.blur > 0) { GaussianBlur(img_scaled, weight, cv::Size(opt.blur, opt.blur), 0, 0); }



	// invert 0.003ms
	bitwise_not(weight, weight);
	//weight /= kWeightDivisor;

	weight.convertTo(weight_float, CV_32F);

	//imshow(debugWindow,weight);
	// set to zero 0.0018ms
	if (out_sum.rows != img_scaled.rows || out_sum.cols != img_scaled.cols)
	{
		out_sum = cv::Mat::zeros(img_scaled.rows, img_scaled.cols, CV_32F);
	}
	out_sum = 0.0f;

}


cv::Point Timm::post_process()
{
	//-- Find the maximum point 0.015 ms
	cv::Point max_point;
	double max_val = 0.0;
	cv::minMaxLoc(out, NULL, &max_val, NULL, &max_point);

	//-- Flood fill the edges 0.05 ms
	if (opt.postprocess_threshold < 1.0f)
	{
		//float floodThresh = computeDynamicThreshold(out, 1.5);
		float flood_threshold = max_val * opt.postprocess_threshold;
		cv::threshold(out, floodClone, flood_threshold, 0.0f, cv::THRESH_TOZERO);


		//if (mask.rows == 0)
		{
			mask = cv::Mat(floodClone.rows, floodClone.cols, CV_8U, 255);
		}
		floodKillEdges(mask, floodClone);

		// redo max
		cv::minMaxLoc(out, NULL, &max_val, NULL, &max_point, mask);
	}

	#ifdef _WIN32
	_ReadWriteBarrier(); 
	#endif
	return max_point;
}

void Timm::prepare_data()
{
	//////// prepare gradients vector 0.025ms /////////// 
	gradients.clear();

	auto cols = gradient_x.cols;
	auto gx_p = gradient_x.ptr<float>(0);
	auto gy_p = gradient_y.ptr<float>(0);
	float c_out = 0.0f;
	const size_t n_floats = simd_width / (8 * sizeof(float));
	simd_data.resize(n_floats * 4);
	size_t k = 0;
	for (size_t y = 0; y < gradient_x.rows; y++)
	{
		for (size_t x = 0; x < gradient_x.cols; x++)
		{
			size_t idx = y * cols + x;
			float gx = gx_p[idx];
			float gy = gy_p[idx];

			if ((gx != 0.0f || gy != 0.0f))
			{
				simd_data[k + 0 * n_floats] = x;
				simd_data[k + 1 * n_floats] = y;
				simd_data[k + 2 * n_floats] = gx;
				simd_data[k + 3 * n_floats] = gy;

				k++;
				if (k == n_floats)
				{
					for (float d : simd_data) { gradients.push_back(d); }
					k = 0;
				}
			}
		}
	}
}


void Timm::floodKillEdges(cv::Mat& mask, cv::Mat &mat)
{
	rectangle(mat, cv::Rect(0, 0, mat.cols, mat.rows), 255);

	std::queue<cv::Point> todo;
	todo.push(cv::Point(0, 0));
	while (!todo.empty())
	{
		cv::Point p = todo.front();
		todo.pop();
		if (mat.at<float>(p) == 0.0f)
		{
			continue;
		}
		// add in every direction
		cv::Point np(p.x + 1, p.y); // right
		if (inside_mat(np, mat)) todo.push(np);

		np.x = p.x - 1; np.y = p.y; // left
		if (inside_mat(np, mat)) todo.push(np);

		np.x = p.x; np.y = p.y + 1; // down
		if (inside_mat(np, mat)) todo.push(np);

		np.x = p.x; np.y = p.y - 1; // up
		if (inside_mat(np, mat)) todo.push(np);

		// kill it
		mat.at<float>(p) = 0.0f;
		mask.at<uchar>(p) = 0;
	}
}


void Timm::imshow_debug(cv::Mat& img, std::string debug_window)
{
	if (img.depth() == CV_32F)
	{
		double min = 0.0, max = 0.0;
		cv::minMaxIdx(img, &min, &max);
		debug_img1 = (img - min) / (max - min);
	}
	else//if(img.depth() == CV_8U)
	{
		debug_img1 = img;
	}
	cv::resize(debug_img1, debug_img2, cv::Size(300, 300));
	imshow(debug_window, debug_img2);
}

///////////////////// original code Tristan Hume, 2012  https://github.com/trishume/eyeLike ///////////////////// 
void Timm::testPossibleCentersFormula(int x, int y, const cv::Mat &weight, double gx, double gy, cv::Mat &out) {
	bool kEnableWeight = true;
	float kWeightDivisor = 1.0f;
	// for all possible centers
	for (int cy = 0; cy < out.rows; ++cy) {
		float *Or = out.ptr<float>(cy);
		const unsigned char *Wr = weight.ptr<unsigned char>(cy);
		for (int cx = 0; cx < out.cols; ++cx) {
			if (x == cx && y == cy) {
				continue;
			}
			// create a vector from the possible center to the gradient origin
			double dx = x - cx;
			double dy = y - cy;
			// normalize d
			double magnitude = sqrt((dx * dx) + (dy * dy));
			dx = dx / magnitude;
			dy = dy / magnitude;
			double dotProduct = dx * gx + dy * gy;
			dotProduct = std::max(0.0, dotProduct);
			// square and multiply by the weight
			if (kEnableWeight) {
				Or[cx] += dotProduct * dotProduct * (Wr[cx] / kWeightDivisor);
			}
			else {
				Or[cx] += dotProduct * dotProduct;
			}
		}
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

float Timm::kernel_orig(float cx, float cy, const cv::Mat& gradientX, const cv::Mat& gradientY)
{
	size_t cols = gradientX.cols;
	auto gx_p = gradientX.ptr<float>(0);
	auto gy_p = gradientY.ptr<float>(0);
	float c_out = 0.0f;
	for (size_t y = 0; y < gradientX.rows; y++)
	{
		for (size_t x = 0; x < gradientX.cols; x++)
		{
			size_t idx = y * cols + x;
			float gx = gx_p[idx];
			float gy = gy_p[idx];

			if (gx != 0.0f || gy != 0.0f)
			{
				float dx = x - cx;
				float dy = y - cy;

				// normalize d
				float magnitude = (dx * dx) + (dy * dy);
				magnitude = sqrt(magnitude); // really needed ?

				dx = dx / magnitude;
				dy = dy / magnitude;

				float dotProduct = dx * gx + dy * gy;
				dotProduct = std::max(0.0f, dotProduct);

				c_out += dotProduct * dotProduct;
			}
		}
	}

	return c_out;
}

