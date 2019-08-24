#ifndef PUPIL_TRACKER_H
#define PUPIL_TRACKER_H

#include "timm_two_stage.h"

#include <atomic>
#include <Eigen/Eigen>

class PupilTracker
{
protected :

	Timm_two_stage timm;

	std::vector<std::array<float, 4> > timings_vector;

	enum enum_parameter_settings
	{
		SETTINGS_DEFAULT,
		SETTINGS_LPW,
		SETTINGS_FUTURE1,
		SETTINGS_FUTURE2,
		SETTINGS_FUTURE3,
		SETTINGS_FROM_FILE,
	};

	using options_type = typename Timm_two_stage::options;
	using params_type = std::array<double, 11>;

	options_type decode_genom(Eigen::VectorXf params);
	options_type load_parameters(enum_parameter_settings s);

	// allowed sizes for the kernels
	const std::array<float, 5> sobel_kernel_sizes{ -1, 1, 3, 5, 7 };
	const std::array<float, 16> blur_kernel_sizes{ 0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29 };

	// helper functions for the fltk gui
	params_type  set_params (options_type opt);
	options_type set_options(params_type params);

	options_type opt;
	std::array<double, 11> params;
	
public :

	PupilTracker();

	void run(const cv::Mat& input, cv::Mat& output);
};

#endif // PUPIL_TRACKER_H

