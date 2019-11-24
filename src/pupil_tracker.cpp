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

void PupilTracker::run(const cv::Mat& input, cv::Mat& output)
{
	impl->run(input, output);
}

