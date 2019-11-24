#ifndef PUPIL_TRACKER_H
#define PUPIL_TRACKER_H

#include <opencv2/core/mat.hpp>

class PupilTrackerImpl;

class PupilTracker
{
	PupilTrackerImpl* impl;
	
public :

	PupilTracker();

	~PupilTracker();

	void run(const cv::Mat& input, cv::Mat& output);
};

#endif // PUPIL_TRACKER_H

