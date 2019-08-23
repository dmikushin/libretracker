#include "cv_camera_control.h"

void Camera_control::change_resolution(int w, int h)
{
	if (cam == nullptr) { return; }

	int id = 0;
	if (cam->isOpened())
	{
		cam->release();
		cam->open(cam->get_index());
		cam->set(cv::CAP_PROP_FRAME_WIDTH, w);
		cam->set(cv::CAP_PROP_FRAME_HEIGHT, h);

		// probe cam and read real width and height
		cv::Mat img;
		if (cam->read(img))
		{
			cam_width = cam->get(cv::CAP_PROP_FRAME_WIDTH);
			cam_height = cam->get(cv::CAP_PROP_FRAME_HEIGHT);
			sg.update_widgets();
		}
	}
}

void Camera_control::setup(std::shared_ptr<Camera> camera, int x, int y, int w, int h, const char* title)
{
	cam = camera;

	sg = Simple_gui(x, y, w, h, title);

	sg.add_separator_box("control (auto)focus, exposure and gain:");
	sg.add_slider("focus:", focus, 0, 255, 1);
	sg.add_slider("exposure:", exposure, -20, 20, 0.01);
	sg.add_slider("gain:", gain, 0, 255, 1);

	sg.add_button("autofocus on", [&]() {cam->set(cv::CAP_PROP_AUTOFOCUS, 1); }, 2, 0);
	sg.add_button("autofocus off", [&]() {cam->set(cv::CAP_PROP_AUTOFOCUS, 0); }, 2, 1);

	sg.add_separator_box("try to set camera resolution to common values:");
	sg.add_button("320x240", [&]() { change_resolution(320, 240); }, 4, 0);
	sg.add_button("640x480", [&]() { change_resolution(640, 480); }, 4, 1);
	sg.add_button("800x600", [&]() { change_resolution(800, 600); }, 4, 2);
	sg.add_button("1280x1024", [&]() { change_resolution(1280, 1024); }, 4, 3);
	sg.add_separator_box("manual resolution setting:");
	sg.add_slider("width :", cam_width, 0, 1920, 10);
	sg.add_slider("height:", cam_height, 0, 1080, 10);
	sg.add_button("try to set selected resolution", [&]()
	{
		change_resolution(cam_width, cam_height);
	});

	//sg.add_slider("window size:", params[2]);
}

void Camera_control::update()
{
	if (nullptr == cam) { return; }
	
	sg.update();
	
	if (focus != old_focus)
	{
		old_focus = focus;
		cam->set(cv::CAP_PROP_FOCUS, focus);
	}

	if (exposure != old_exposure)
	{
		old_exposure = exposure;
		cam->set(cv::CAP_PROP_EXPOSURE, exposure);
	}

	if (gain != old_gain)
	{
		old_gain = gain;
		cam->set(cv::CAP_PROP_GAIN, gain);
	}
}



#ifdef __TEST_THIS_MODULE__

#include <iostream>

#ifdef _DEBUG
#pragma comment(lib, "opencv_world343d.lib")
#pragma comment(lib, "fltk14/bin/lib/Debug/fltkd.lib")
#else
#pragma comment(lib, "opencv_world343.lib")
#pragma comment(lib, "fltk14/bin/lib/Release/fltk.lib")
#endif

#ifdef _WIN32
#pragma comment(lib, "wsock32.lib")
#pragma comment(lib, "comctl32.lib")
#endif 

#include "../../timer_hd.h"

using namespace std;
using namespace cv;

void test_module()
{
	setUseOptimized(true);

	Mat img;
	
	auto cam = make_shared<Camera>(0);

	//cap->set(cv::CAP_PROP_AUTOFOCUS, 0);
	//cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0);
	//cam->set(cv::CAP_PROP_EXPOSUREPROGRAM, 0);

	Camera_control cam_control;
	
	cam_control.setup(cam, 20, 20, 400, 400, "My Cam");

	Timer timer(50);
	int counter = 0;
	while (true)
	{
		timer.tick();
		if (cam->read(img))
		{
			imshow("frame", img);
		}

		cam_control.update();
		counter++;

		timer.tock();
	}

	cv::destroyAllWindows();

	return 1;
}

#endif