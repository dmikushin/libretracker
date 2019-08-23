#pragma once

#include <memory>
#include <thread>
#include <string>
#include <atomic>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "cv_camera_control.h"


// multithreaded capture variables
class Threaded_capture
{
protected:
	std::thread capture_thread;
	std::mutex mx;
	std::string cam_name;
	std::shared_ptr<Camera> camera;
	cv::Mat frame;
	std::atomic<bool> is_running;
	void capture_thread_func();
public:
	std::atomic<bool> new_frame;

	void setup(std::shared_ptr<Camera> cam, std::string cam_name_);

	void get_frame(cv::Mat &frame_external)
	{
		mx.lock();
		frame.copyTo(frame_external);
		mx.unlock();
	}

	void stop();
};
