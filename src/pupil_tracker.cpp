#include "pupil_tracker.h"

#include <random>

using namespace cv;
using namespace std;

PupilTracker::PupilTracker()
{
	// generate a random image
	auto img_rand = Mat(480, 640, CV_8UC3);
	randu(img_rand, Scalar::all(0), Scalar::all(255));

	cv::Mat frame;
	cv::Mat frame_gray;
	
	opt = load_parameters(SETTINGS_LPW);
	params = set_params(opt);
}

void PupilTracker::run(const cv::Mat& input, cv::Mat& output)
{
	opt = set_options(params);
	timm.set_options(opt);

	// Apply the classifier to the frame
	if (!input.empty())
	{
		if (opt.blur > 0) { GaussianBlur(input, output, cv::Size(opt.blur, opt.blur), 0); }

		// auto[pupil_pos, pupil_pos_coarse] = ...  (structured bindings available with C++17)
		cv::Point pupil_pos, pupil_pos_coarse;
		std::tie(pupil_pos, pupil_pos_coarse) = timm.pupil_center(output);
		//pupil_pos = timm.stage1.pupil_center(output);
		output = input;
		timm.visualize_frame(output, pupil_pos, pupil_pos_coarse);
	}
}

PupilTracker::options_type PupilTracker::load_parameters(enum_parameter_settings s)
{
	using namespace EL;
	auto param_set = zeros(10);
	options_type opt;

	switch (s)
	{
	case SETTINGS_LPW:
		// optimal parameters for LPW dataset from differential evolution result
		param_set << 0.3692, 1.8457, -0.6424, 0.3509, 0.8782, 1.5772, -0.4682, 0.1485, 0.3652, 0.7056; // median <-- best, final solution !
		opt = decode_genom(param_set);
		break;

	case SETTINGS_DEFAULT:
	default:
		break;
	}

	return opt;
}

PupilTracker::params_type PupilTracker::set_params(options_type opt)
{
	params_type params;
	params[0] = opt.window_width;
	params[1] = opt.stage1.gradient_threshold;
	params[2] = opt.stage2.gradient_threshold;
	params[3] = opt.stage1.sobel;
	params[4] = opt.stage2.sobel;
	params[5] = opt.stage1.postprocess_threshold;
	params[6] = opt.stage2.postprocess_threshold;
	params[7] = opt.stage1.blur;
	params[8] = opt.stage2.blur;
	params[9] = opt.blur;
	params[10] = opt.stage1.down_scaling_width; // assuming for now that this value is identical for both stages
	return params;
}

PupilTracker::options_type PupilTracker::set_options(params_type params)
{
	options_type opt;
	// set options from gui parameter array
	opt.window_width = params[0];

	opt.stage1.gradient_threshold = params[1];
	opt.stage2.gradient_threshold = params[2];

	opt.stage1.sobel = to_closest(params[3], sobel_kernel_sizes);
	opt.stage2.sobel = to_closest(params[4], sobel_kernel_sizes);

	opt.stage1.postprocess_threshold = params[5];
	opt.stage2.postprocess_threshold = params[6];

	opt.stage1.blur = to_closest(params[7], blur_kernel_sizes);
	opt.stage2.blur = to_closest(params[8], blur_kernel_sizes);

	opt.blur = to_closest(params[9], blur_kernel_sizes);

	opt.stage1.down_scaling_width = params[10];
	opt.stage2.down_scaling_width = params[10];
	return opt;
}

PupilTracker::options_type PupilTracker::decode_genom(Eigen::VectorXf params)
{
	// allowed sobel kernel sizes
	array<float, 5> sobel_kernel_sizes{ -1, 1, 3, 5, 7 };

	options_type opt;

	opt.blur = 2 * clip<int>(30 * params[0], 0, 30) + 1;
	opt.window_width = clip<int>(250 * params[3], 50, 250);

	opt.stage1.sobel = to_closest<5>(4 * params[1], sobel_kernel_sizes);
	opt.stage2.sobel = to_closest<5>(4 * params[2], sobel_kernel_sizes);

	opt.stage1.postprocess_threshold = clip<float>(params[4], 0, 1); // 65; // for videos with dark eye lashes
	opt.stage2.postprocess_threshold = clip<float>(params[5], 0, 1); // 65; // for videos with dark eye lashes

	// set params
	opt.stage1.gradient_threshold = clip<float>(255 * params[6], 0, 255);
	opt.stage2.gradient_threshold = clip<float>(255 * params[7], 0, 255);

	// this must be odd numbers
	opt.stage1.blur = 2 * clip<int>(10 * params[8], 0, 10) + 1;
	opt.stage2.blur = 2 * clip<int>(10 * params[9], 0, 10) + 1;

	// for now, this value is hardcoded 
	opt.stage1.down_scaling_width = 85;
	opt.stage2.down_scaling_width = 85;

	return  opt;
}

