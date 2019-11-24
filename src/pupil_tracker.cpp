#include "pupil_tracker.h"
#include "pupil_tracker_impl.h"

PupilTracker::PupilTracker()
{
	impl = new PupilTrackerImpl();
}

PupilTracker::~PupilTracker()
{
	delete impl;
}

void PupilTracker::run(const cv::Mat& image, cv::Point& pupil_pos, cv::Point& pupil_pos_coarse)
{
	impl->run(image, pupil_pos, pupil_pos_coarse);
}

void PupilTracker::annotate(cv::Mat& image, const cv::Point& pupil_pos, const cv::Point& pupil_pos_coarse)
{
	impl->annotate(image, pupil_pos, pupil_pos_coarse);
}

