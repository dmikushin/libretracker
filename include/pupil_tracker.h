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

	void run(const cv::Mat& image, cv::Point& pupil_pos, cv::Point& pupil_pos_coarse);

	void annotate(cv::Mat& image, const cv::Point& pupil_pos, const cv::Point& pupil_pos_coarse);
};

#endif // PUPIL_TRACKER_H

