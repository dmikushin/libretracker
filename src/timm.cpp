#include "timm.h"
#include "timing.h"
#include "eigenlab.h"
#include "filters.h"
#include "eigen_pseudoinverse.h"

#include <Eigen/Eigen>
#include <queue>
#include <iostream>

using namespace std;

Timm::Timm() { }

unsigned int Timm::getSIMDVectorSize()
{
	return 1;
}

float* Timm::getGradients()
{
	if (!gradients.size())
		gradients.resize(scaledWidth() * scaledHeight() * 4);

	return reinterpret_cast<float*>(&gradients[0]);
}

void Timm::pupilCenterKernel(cv::Mat& out_sum, float* gradients, int ngradients)
{ 
	auto data = out_sum.ptr<float>(0);
	for (unsigned int y = 0; y < out_sum.rows; y++)
		for (unsigned int x = 0; x < out_sum.cols; x++)
			data[y * out_sum.cols + x] = kernel(x, y, gradients, ngradients);
}

unsigned int Timm::scaledWidth() const { return img_scaled.cols; }
unsigned int Timm::scaledHeight() const { return img_scaled.rows; }

cv::Point Timm::pupilCenter(const cv::Mat& eye_img)
{
	preprocess(eye_img);

	float* gradients = getGradients();
	int ngradients = prepareData(gradients);

	pupilCenterKernel(out_sum, gradients, ngradients);

	cv::multiply(out_sum, weight_float, out);

	cv::Point result = remap(postprocess(), eye_img.cols);

	return result;
}

float Timm::kernel(int cx, int cy, float* gradients, int ngradients)
{
	float c_out = 0.0f;

	for (int i = 0; i < ngradients; i += 4)
	{
		const float* sd = &gradients[i];

		float x  = sd[0];
		float y  = sd[1];
		float gx = sd[2];
		float gy = sd[3];
		
		float dx = x - cx;
		float dy = y - cy;

		float dotProduct = dx * gx + dy * gy;
		if (dotProduct > 0.0f)
		{
			// normalize d
			float magnitude = (dx * dx) + (dy * dy);
			magnitude = 1.0f / magnitude;
			c_out += dotProduct * dotProduct * magnitude;
		}
	}
		
	return c_out;
}

float Timm::calcDynamicThreshold(const cv::Mat &mat, float stdDevFactor)
{
	cv::Scalar stdMagnGrad, meanMagnGrad;
	cv::meanStdDev(mat, meanMagnGrad, stdMagnGrad);
	float stdDev = stdMagnGrad[0] / sqrt(mat.rows*mat.cols);
	return stdDevFactor * stdDev + meanMagnGrad[0];
}

void Timm::preprocess(const cv::Mat& img)
{
	// down sample to speed up
	cv::resize(img, img_scaled, cv::Size(opt.down_scaling_width, img.rows * float(opt.down_scaling_width) / img.cols));

	// calc the gradients 0.06ms
	// valid inputs to sobel kernel size: -1, 1, 3, 5, 7 
	cv::Sobel(img_scaled, gradient_x, CV_32F, 1, 0, opt.sobel);
	cv::Sobel(img_scaled, gradient_y, CV_32F, 0, 1, opt.sobel);
	
	// compute all the magnitudes 0.01ms
	cv::Mat mags = gradient_x.mul(gradient_x) + gradient_y.mul(gradient_y);
	sqrt(mags, mags); // is sqrt really necessary ??

	//-- Normalize and threshold the gradient
	//compute the threshold 0.004ms
	float gradientThresh = calcDynamicThreshold(mags, opt.gradient_threshold);

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

	// set to zero 0.0018ms
	if (out_sum.rows != img_scaled.rows || out_sum.cols != img_scaled.cols)
	{
		out_sum = cv::Mat::zeros(img_scaled.rows, img_scaled.cols, CV_32F);
	}
	out_sum = 0.0f;
}

cv::Point Timm::postprocess()
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

	return max_point;
}

//////// prepare gradients vector 0.025ms /////////// 
int Timm::prepareData(float* gradients)
{
	unsigned int n_floats = getSIMDVectorSize();

	int ngradients = 0;

	auto cols = gradient_x.cols;
	auto gx_p = gradient_x.ptr<float>(0);
	auto gy_p = gradient_y.ptr<float>(0);
	float c_out = 0.0f;
	std::vector<float> simd_data(n_floats * 4);
	int k = 0;
	for (int y = 0, k = 0; y < gradient_x.rows; y++)
	{
		for (int x = 0; x < gradient_x.cols; x++)
		{
			size_t idx = y * cols + x;
			float gx = gx_p[idx];
			float gy = gy_p[idx];

			if ((gx == 0.0f) && (gy == 0.0f))
				continue;

			simd_data[k + 0 * n_floats] = x;
			simd_data[k + 1 * n_floats] = y;
			simd_data[k + 2 * n_floats] = gx;
			simd_data[k + 3 * n_floats] = gy;

			k++;
			if (k == n_floats)
			{
				for (float d : simd_data)
					gradients[ngradients++] = d;
				k = 0;
			}
		}
	}
	
	return ngradients;
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
		if (insideMat(np, mat)) todo.push(np);

		np.x = p.x - 1; np.y = p.y; // left
		if (insideMat(np, mat)) todo.push(np);

		np.x = p.x; np.y = p.y + 1; // down
		if (insideMat(np, mat)) todo.push(np);

		np.x = p.x; np.y = p.y - 1; // up
		if (insideMat(np, mat)) todo.push(np);

		// kill it
		mat.at<float>(p) = 0.0f;
		mask.at<uchar>(p) = 0;
	}
}

