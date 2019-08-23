#pragma once

#include <opencv2/opencv.hpp>
#include <memory>

#include "simple_gui_fltk.h"

// very simple Wrapper Class cv::VideoCapture - only needed to store the camera device index
class Camera : public cv::VideoCapture
{
protected:
	int index = 0;
public:
	Camera(int idx) : cv::VideoCapture(idx)
	{
		index = idx;
	}
	int get_index() { return index; }
};

class Camera_control
{
protected:
	double focus = 0.0, old_focus = 0.0;
	double exposure = 0.0, old_exposure = 0.0;
	double gain = 0.0, old_gain = 0.0;
	Simple_gui sg;
	std::shared_ptr<Camera> cam;
	double cam_width=0, cam_height=0;

	//void change_resolution(shared_ptr<Camera> cap, int w, int h)
	void change_resolution(int w, int h);

public:	
	void setup(std::shared_ptr<Camera> camera, int x, int y, int w, int h, const char* title);
	void update();
};