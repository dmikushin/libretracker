#ifndef TIMM_H
#define TIMM_H

#include <array>
#include <algorithm>
#include <vector>
#include <opencv2/imgproc.hpp>

// the template parameter instrSet specifies the vector register bit width
// e.g. 512 for AVX512, 256 for AVX2 and 128 for SSE
class Timm
{
	// this vector stores sequential chunks of floats for x, y, gx, gy
	std::vector<float> gradients;
	std::vector<float> simd_data;

	cv::Mat gradient_x;
	cv::Mat gradient_y;
	cv::Mat img_scaled;
	cv::Mat mags;
	cv::Mat weight;
	cv::Mat weight_float;
	cv::Mat out_sum;
	cv::Mat out;
	cv::Mat floodClone;
	cv::Mat mask;

	void preprocess(const cv::Mat& img);
	cv::Point postprocess();

	inline cv::Point remap(cv::Point p, float original_width)
	{
		float s = original_width / float(opt.down_scaling_width);
		int x = round( s * p.x );
		int y = round( s * p.y );
		return cv::Point(x, y);
	}

	float calcDynamicThreshold(const cv::Mat &mat, float stdDevFactor);

	int prepareData(float* ngradients);

	inline bool insideMat(cv::Point p, const cv::Mat &mat)
	{
		return p.x >= 0 && p.x < mat.cols && p.y >= 0 && p.y < mat.rows;
	}

	// Compute a floodfiling mask.
	void floodKillEdges(cv::Mat& mask, cv::Mat &mat);
	
protected :

	virtual unsigned int getSIMDVectorSize();

	virtual float* getGradients();

	virtual void pupilCenterKernel(cv::Mat& out_sum, float* gradients, int ngradients);

	virtual float kernel(float cx, float cy, float* gradients, int ngradients);

public :

	unsigned int scaledWidth() const;
	unsigned int scaledHeight() const;

	Timm();

	// Algorithm Parameters 
	struct options
	{
		int down_scaling_width = 85;
		int blur = 5; // gaussan blurr kernel size. must be an odd number > 0
		int sobel = 5; // must be either -1, 1, 3, 5, 7
		float gradient_threshold = 50.0f; //50.0f;
		float postprocess_threshold = 0.97f;
	}
	opt;

	// estimates the pupil center
	// inputs: eye image
	virtual cv::Point pupilCenter(const cv::Mat& eye_img);
};

#endif // TIMM_H

