#ifndef TIMM_TWO_STAGE_H
#define TIMM_TWO_STAGE_H

#if ENABLE_GPU
#include "timm_gpu.h"
#else
#include "timm_multihreaded.h"
#endif
#include "helpers.h"

class Timm_two_stage
{
	cv::Mat frame_gray_windowed;
	cv::Mat frame_color;

public :

	Timm_two_stage() { }

	struct options
	{
		using timm_options = typename Timm::options;
		int blur = 0;
		int window_width = 150;
		timm_options stage1; // coarse pupil center estimation stage
		timm_options stage2; // fine, windowed pupil center estimation stage
	}
	opt;

#if ENABLE_GPU
	TimmGPU stage1;
	TimmGPU stage2;
#else
	TimmMultithreaded stage1;
	TimmMultithreaded stage2;
#endif

	void set_options(options o)
	{
		opt = o;
		stage1.opt = opt.stage1;
		stage2.opt = opt.stage2;

		// If the window width is smaller than the down scaling width,
		// make the down_scaling_width equal to the window width to save processing time.
		stage2.opt.down_scaling_width = std::min(stage2.opt.down_scaling_width, opt.window_width);
	}

	// two stages: coarse estimation and local refinement of pupil center
	std::tuple<cv::Point, cv::Point> pupilCenter(cv::Mat& frame_gray)
	{
		//-- Find Eye Centers
		cv::Point pupil_pos_coarse = stage1.pupilCenter(frame_gray);
		
		auto rect = fit_rectangle(frame_gray, pupil_pos_coarse, opt.window_width);
		frame_gray_windowed = frame_gray(rect);
		cv::Point pupil_pos = stage2.pupilCenter(frame_gray_windowed);
		
		pupil_pos.x += rect.x;
		pupil_pos.y += rect.y;
		return std::tie(pupil_pos, pupil_pos_coarse);
	}

private :

	///////// visualisation stuff ///////////

	template<class T> void draw_cross(cv::Mat& img, T p, int w, cv::Scalar col = cv::Scalar(255, 255, 255))
	{
		float w2 = 0.5*w;
		cv::line(img, cv::Point(p.x - w2, p.y), cv::Point(p.x + w2, p.y), col);
		cv::line(img, cv::Point(p.x, p.y - w2), cv::Point(p.x, p.y + w2), col);
	}


	// fit a rectangle with center c and half-width w into a given image
	cv::Rect fit_rectangle(cv::Mat frame, cv::Point2f c, int w)
	{
		int w2 = w; // half width of windows
		w2 = clip<int>(w2, 0, 0.5*frame.cols);
		w2 = clip<int>(w2, 0, 0.5*frame.rows);

		int x = c.x; x = clip<float>(x, w2, frame.cols - w2);
		int y = c.y; y = clip<float>(y, w2, frame.rows - w2);
		return cv::Rect(x - w2, y - w2, 2 * w2, 2 * w2);
	}

public :

	void visualize_frame(cv::Mat& frame, cv::Point2f pupil_pos, cv::Point2f pupil_pos_coarse)
	{
		cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
		auto rect = fit_rectangle(frame, pupil_pos_coarse, opt.window_width);

		// draw local processing rectangle
		cv::rectangle(frame, rect, cv::Scalar(0, 0, 155));
		
		// draw eye center coarse
		//circle(frame, pupil_pos_coarse, 2, cv::Scalar(255, 0, 0), 1);
		
		// draw eye center fine
		draw_cross(frame, pupil_pos, 7, cv::Scalar(0, 255, 0));
	}
};

#endif // TIMM_TWO_STAGE_H

