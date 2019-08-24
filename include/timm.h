#ifndef TIMM_H
#define TIMM_H

#include <array>
#include <algorithm>
#include <vector>
#include <opencv2/imgproc.hpp>

// needed to access Vector Extension Instructions
#ifdef __arm__
#include <arm_neon.h>
#endif

#include "dependencies.h"
#include "instrset.h"
#include "aligned_allocator.h"

// the template parameter instrSet specifies the vector register bit width
// e.g. 512 for AVX512, 256 for AVX2 and 128 for SSE
class Timm
{
protected :

	agner::InstrSet instrSet;
	size_t n_floats;
	int n_threads;

	// optimised for SIMD: 
	// this vector stores sequential chunks of floats for x, y, gx, gy
	std::vector<float, AlignedAllocator<float, Alignment::AVX512> > gradients;
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
	cv::Mat debug_img1;
	cv::Mat debug_img2;

public :

	std::array<bool, 4> debug_toggles{ false, false, false , false };

	// for timing measurements
	float measure_timings[2] = { 0, 0 };
	
	Timm();

	// Algorithm Parameters 
	struct options
	{
		int down_scaling_width = 85;
		int blur = 5; // gaussan blurr kernel size. must be an odd number > 0
		int sobel = 5; // must be either -1, 1, 3, 5, 7
		float gradient_threshold = 50.0f; //50.0f;
		float postprocess_threshold = 0.97f;
	} opt;

	// estimates the pupil center
	// inputs: eye image, reagion of interest (rio) and an optional window name for debug output
	cv::Point pupil_center(const cv::Mat& eye_img);


protected:

	void pre_process(const cv::Mat& img);
	cv::Point post_process();


	inline cv::Point undo_scaling(cv::Point p, float original_width)
	{
		float s = original_width / float(opt.down_scaling_width);
		int x = round( s * p.x );
		int y = round( s * p.y );
		return cv::Point(x, y);
	}

	float calc_dynamic_threshold(const cv::Mat &mat, float stdDevFactor);

	void imshow_debug(cv::Mat& img, std::string debug_window);

	void prepare_data();

	inline bool inside_mat(cv::Point p, const cv::Mat &mat)
	{
		return p.x >= 0 && p.x < mat.cols && p.y >= 0 && p.y < mat.rows;
	}

	// TODO: optimize !
	// returns a mask
	void floodKillEdges(cv::Mat& mask, cv::Mat &mat);

	///////////////////// original code Tristan Hume, 2012  https://github.com/trishume/eyeLike ///////////////////// 
	void testPossibleCentersFormula(int x, int y, const cv::Mat &weight, double gx, double gy, cv::Mat &out);
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	float kernel_orig(float cx, float cy, const cv::Mat& gradientX, const cv::Mat& gradientY);

#ifdef __arm__
	inline float kernel_op_arm128(float cx, float cy, const float* sd)
	{
		// is it faster with "static" ?
		static const float32x4_t zero = vdupq_n_f32(0.0f); 

		// set cx value into vector (vdupq_n_f32 == Load all lanes of vector to the same literal value)
		//float32x4_t cx_4 = vdupq_n_f32(cx);
		//float32x4_t cy_4 = vdupq_n_f32(cy);

		// loading memory into vector registers
		float32x4_t dx_in = vld1q_f32(sd);
		float32x4_t dy_in = vld1q_f32(sd +  4);
		float32x4_t gx_in = vld1q_f32(sd +  8);
		float32x4_t gy_in = vld1q_f32(sd + 12);

		// calc the difference vector
		//dx_in = vsubq_f32(dx_in, cx_4);
		//dy_in = vsubq_f32(dy_in, cy_4);
		dx_in = vmulq_n_f32(dx_in, cx);
		dy_in = vmulq_n_f32(dy_in, cy);

		// calc the dot product	
		float32x4_t tmp1 = vmulq_f32(dx_in, dx_in);
		float32x4_t tmp2 = vmulq_f32(dy_in, dy_in);
		tmp1 = vaddq_f32(tmp1, tmp2);

		// now cals the reciprocal square root
		tmp1 = vrsqrteq_f32(tmp1);

		// now normalize by multiplying
		dx_in = vmulq_f32(dx_in, tmp1);
		dy_in = vmulq_f32(dy_in, tmp1);

		// now calc the dot product with the gradient
		tmp1 = vmulq_f32(dx_in, gx_in);
		tmp2 = vmulq_f32(dy_in, gy_in);
		tmp1 = vaddq_f32(tmp1, tmp2);

		// now calc the maximum // does this really help ???
		tmp1 = vmaxq_f32(tmp1, zero);
		//tmp1 = vmaxq_n_f32(tmp1, 0.0f); // this instruction sadly does not exist
		
		// multiplication 
		tmp1 = vmulq_f32(tmp1, tmp1);

		//return vaddvq_f32(tmp1);
	
		// https://pmeerw.net/blog/programming/neon1.html
		// accumulate four quadword floats
		static const float32x2_t f0 = vdup_n_f32(0.0f);
		return vget_lane_f32(vpadd_f32(f0, vget_high_f32(tmp1) + vget_low_f32(tmp1)), 1);
	}
#endif // __arm__
#ifdef SSE41_ENABLED
	float kernel_op_sse(float cx, float cy, const float* sd);
#endif
#ifdef AVX_ENABLED
	float kernel_op_avx(float cx, float cy, const float* sd);
#endif
#ifdef AVX2_ENABLED
	float kernel_op_avx2(float cx, float cy, const float* sd);
#endif
#ifdef AVX512_ENABLED
	float kernel_op_avx512(float cx, float cy, const float* sd);
#endif
	
	inline float kernel_op(float cx, float cy, const float* sd)	
	{
		
		float x  = sd[0];
		float y  = sd[1];
		float gx = sd[2];
		float gy = sd[3];
		
		float dx = x - cx;
		float dy = y - cy;

		// normalize d
		float magnitude = (dx * dx) + (dy * dy);

		// very time consuming.. ? // with this 28.8 ms
		//magnitude = sqrt(magnitude); 
		//dx = dx / magnitude;
		//dy = dy / magnitude;

		//magnitude = fast_inverse_sqrt_quake(magnitude); // with this: 26 ms.
		//magnitude = fast_inverse_sqrt_around_one(magnitude); // not working .. 

#ifdef _WIN32 // currently fast_inverse_sqrt is only defined for win32
		fast_inverse_sqrt(&magnitude, &magnitude); // MUCH FASTER !
#else
		magnitude = 1.0f / sqrt(magnitude);
#endif
		dx = dx * magnitude;
		dy = dy * magnitude;

		float dotProduct = dx * gx + dy * gy;
		dotProduct = std::max(0.0f, dotProduct);

		return dotProduct * dotProduct;
	}

	float kernel(float cx, float cy);
};

#endif // TIMM_H

