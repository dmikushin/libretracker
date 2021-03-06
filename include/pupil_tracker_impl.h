#ifndef PUPIL_TRACKER_IMPL_H
#define PUPIL_TRACKER_IMPL_H

#include "timm_two_stage.h"

#include <Eigen/Eigen>

class PupilTrackerImpl
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

	PupilTrackerImpl();

	void run(const cv::Mat& image, cv::Point& pupil_pos, cv::Point& pupil_pos_coarse);

	void annotate(cv::Mat& image, const cv::Point& pupil_pos, const cv::Point& pupil_pos_coarse);
};

#endif // PUPIL_TRACKER_IMPL_H

