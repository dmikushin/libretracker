#include "eyetracking_speller.h"


static void mouse_callback(int event, int x, int y, int, void* user_data)
{
	using namespace cv;

	Eyetracking_speller* ptr = static_cast<Eyetracking_speller*>(user_data);

	/*
	bool eye_button_up = false, eye_button_down = false;

	if (event == EVENT_LBUTTONUP) { eye_button_up = true; eye_button_down = false; }
	if (event == EVENT_LBUTTONDOWN) { eye_button_down = true; eye_button_up = false; }
	*/

	//ptr->set_mouse(x, y, eye_button_up, eye_button_down);
	ptr->set_mouse(x, y, false, false);
	
	if (event == EVENT_LBUTTONUP) { cout << "mauspos: " << x << "\t" << y << endl; }
}



void Eyetracking_speller::setup(enum_simd_variant simd_width)
{
	
	Pupil_tracking::setup(simd_width, PUPIL_TRACKING_PUREST);

	using namespace cv;
	using namespace EL;


	// TODO: load previous calibration matrix from file

	speller.setup();
	calibration.setup();

	/*
	// setup gradient based pupil capture
	opt = load_parameters(SETTINGS_LPW);
	timm.set_options(opt);

	// prepare gui for gradient based pupil capture
	params = set_params(opt);
	//setup_gui();
	*/

	// GUI
	//sg = Simple_gui(min(Fl::w() - 200, 1420), 180, 400, 600);
	sg = Simple_gui(20, 60, 400, 620);

	sg.add_separator_box("1. adjust canvas size and AR-marker tracking:");
	sg.add_slider("canvas width", gui_param_w, 640, 5000, 10, "Change the width of the canvas to fit your monitor size. Make sure that all screen-tracking markers fit into the field of view of the scene camera.");
	sg.add_slider("canvas height", gui_param_h, 480, 3000, 10, "Change the height of the canvas.");
	sg.add_slider("AR marker size", gui_param_marker_size, 40, 400, 10, "Size of the AR markers in pixel. Larger values increase the robustness of marker tracking, but reduce the available screen space for e.g. the speller application.");
	// sg.add_button("use enclosed markers", [&]() {calibration.ar_canvas.setup(true); },1,0,"use enclosed markers. corners jitter less, but markers need to be larger.");
	// sg.add_slider("detection size", calibration.ar_canvas.min_marker_size, 0.005, 0.1, 0.001, "minimum detection size of a marker (in percent of total image area)");

	sg.add_separator_box("2. adjust cameras:");
	sg.add_button("swap cameras", [&]() { auto tmp = scene_camera; scene_camera = eye_camera; eye_camera = tmp; }, 3, 0, "swap eye- and scene camera.");
	sg.add_button("eye-cam", [&]() { eye_cam_controls.setup(eye_camera, 20, 20, 400, 400, "Eye-Camera Controls"); }, 3, 1, "adjust the eye-camera settings.");
	sg.add_button("scene-cam", [&]() { scene_cam_controls.setup(scene_camera, 20, 20, 400, 400, "Scene-Camera Controls"); }, 3, 2, "adjust the scene-camera settings.");

	sg.add_separator_box("3. select Pupil-Tracking algorithm:");
	sg.add_radio_button("Timm's algorithm", [&,s = simd_width]() { Pupil_tracking::setup(s, PUPIL_TRACKING_TIMM); });
	sg.add_radio_button("PuRe (for research only!)", [&, s = simd_width]() {Pupil_tracking::setup(s, PUPIL_TRACKING_PURE); });
	auto button = sg.add_radio_button("PuReST (for research only!)", [&, s = simd_width]() {Pupil_tracking::setup(s, PUPIL_TRACKING_PUREST); });
	button->value(true);
	sg.add_button("adjust settings", [&]() { pupil_tracker->show_gui(); }, 1, 0);


	/*
	// TODO TODO TODO !!
	sg.add_separator_box("camera calibration (not implemented yet):");
	sg.add_button("scene camera", [&]() { state = STATE_CALIBRATION_SCENE_CAM; }, 2, 0);
	sg.add_button("eye camera", [&]() { state = STATE_CALIBRATION_EYE_CAM; }, 3, 1);
	sg.add_button("save", [&]() {}, 3, 2);
	*/
	
	sg.add_separator_box("4. calibrate the eyetracker:");
	// TODO sg.add_slider("n poly features", []() {}, 4, 4, 10, "");
	sg.add_button("5 point",	[&]() { grab_focus("screen"); calibration.setup(5); state = STATE_CALIBRATION; }, 3, 0, "perform a 5-point calibration.");
	sg.add_button("9 point",	[&]() { grab_focus("screen"); calibration.setup(9); state = STATE_CALIBRATION; }, 3, 1, "perform a 9-point calibration. this takes a bit longer, but usually increases calibration accuracy.");
	sg.add_button("visualize",	[&]() { state = STATE_CALIBRATION; calibration.state = Calibration::STATE_VISUALIZE_CALIBRATION; }, 3, 2, "Visualize the calibration result. Here, you can also try to optimize the polynomial 2d-to-2d mapping by changing the number of polynomial features.");
	
	sg.add_separator_box("5. validate the calibration (optional):");
	double n_validation_points = 5;
	sg.add_slider("validation points", n_validation_points, 4, 20, 1,"Select the number of validation points.");
	sg.add_slider("randomness [px]", n_validation_points, 0, 50, 1, "Select here, how much the validation points randomly deviate from the default validation positions.");
	sg.add_button("validate",	[&]() { grab_focus("screen"); calibration.setup_validation(); calibration.state = Calibration::STATE_VALIDATION;  }, 3, 0, "check the calibration by testing additional points. (optional)");
	sg.add_button("visualize", [&]() { calibration.state = Calibration::STATE_VISUALIZE_VALIDATION;  }, 3, 1, "Visalizes the results of the validation.");
	sg.add_button("fix offset", [&]() { calibration.fix_offset();  }, 3, 2, "remove a potential systematic offset found after validation.");


	sg.add_separator_box("6. run modules and adjust jitter filter:");
	sg.add_slider("smoothing", filter_smoothing, 0, 1, 0.01, "adjust the amount of smoothing of the jitter filter (double exponential filter). Larger values reduce jitter, but introduce noticable lag. this lag can partially compensated increasing the predictive value.");
	sg.add_slider("predictive", filter_predictive, 0, 1, 0.001, "The predictive component can partially comensate the lag introduced by smoothing. Large values can cause overshooting and damped oscillations of the filter.");
	sg.add_button("run speller", [&]() { grab_focus("screen"); state = STATE_RUNNING; }, 3, 0);
	sg.add_button("stream to client", [&]() { run_ssvep(); }, 3, 1);
	sg.add_button("quit", [&]() { sg.hide(); Fl::check(); is_running = false; }, 3, 2);


	sg.finish();
	sg.show();

	//#ifndef HAVE_OPENGL
	//if (flags & CV_WINDOW_OPENGL) CV_ERROR(CV_OpenGlNotSupported, "Library was built without OpenGL support");
	//#else
	//setOpenGlDrawCallback("windowName", glCallback);

	namedWindow("eye_cam");
	namedWindow("screen");// , WINDOW_OPENGL | WINDOW_AUTOSIZE);

	// place the main window to the right side of the options gui
	moveWindow("screen", 450, 10);
	resizeWindow("screen", w, h);
	
	moveWindow("eye_cam", 20, 700);

	// uncomment to simulate gaze using mouse 
	cv::setMouseCallback("screen", mouse_callback, this);

	// code for special calibration marker
	calibration.setup(4);
}



void Eyetracking_speller::draw_instructions()
{
	using namespace cv;
	int mb = calibration.ar_canvas.marker_size;
	// auto screen_center = Point2f(0.5f*img_screen.cols, 0.5f*img_screen.rows);

	calibration.ar_canvas.draw(img_screen, 0, 0, w, h);

	int y = mb + 25;
	auto print_txt = [&](const char * t)
	{
		putText(img_screen, t, Point2i(mb, y), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 170, 0), 2);
		y += 35;
	};

	print_txt("1. Adjust the size of the window such that the scene camera");
	print_txt("   can see all AR markers at your current distance.");
	print_txt("2. Calibrate the eyetracker, then validate (optional).");
	print_txt("3. Press the run button to start the speller.");

	draw_preview(frame_scene_cam, img_screen);

	imshow("screen", img_screen);

}

void Eyetracking_speller::draw_speller(bool ssvep)
{
	using namespace cv;
	int mb = calibration.ar_canvas.marker_size;

	calibration.ar_canvas.draw(img_screen, 0, 0, w, h);
	// ********************************************************
	// draw keyboard and handle events
	if (ssvep)
	{
		speller.draw_keyboard_ssvep(img_screen, 0, mb, w, h - 2 * mb, mb, p_projected.x, p_projected.y, eye_button_up);
	}
	else
	{
		speller.draw_keyboard(img_screen, 0, mb, w, h - 2 * mb, mb, p_projected.x, p_projected.y, eye_button_up);
	}

	// draw gaze point after coordinate transformation		
	circle(img_screen, p_projected, 8, Scalar(255, 0, 255), 4);

	// draw gaze point after calibration
	circle(frame_scene_cam, p_calibrated, 4, Scalar(255, 0, 255), 2);

	// imshow("scene_cam", frame_scene_cam_copy);
	float scaling = float(mb) / float(frame_scene_cam.rows);
	draw_preview(frame_scene_cam, img_screen, scaling, -1, img_screen.rows - mb);
}

void Eyetracking_speller::draw()
{
	using namespace cv;

	// clear canvas
	img_screen_background.copyTo(img_screen);

	// visualize marker detection // TODO: move to subfunctions for different states
	calibration.ar_canvas.draw_detected_markers(frame_scene_cam);

	switch (state)
	{
	case STATE_INSTRUCTIONS:	draw_instructions();  break;
	case STATE_CALIBRATION:		calibration.draw(frame_scene_cam, img_screen); break;
	case STATE_RUNNING:			draw_speller(); break; 
	default: break;

	}

	pupil_tracker->draw(frame_eye_cam);
	imshow("eye_cam", frame_eye_cam);
	imshow("screen", img_screen);
}




// update function for all steps from camera setup, calibration, validation and single threaded speller
// multithreaded eyetracking / speller use a different function
void Eyetracking_speller::update()
{
	using namespace cv;


	// update values that might have been changed via GUI sliders
	gaze_filter_x.set_params(1.0 - filter_smoothing, filter_predictive);
	gaze_filter_y.set_params(1.0 - filter_smoothing, filter_predictive);

	// if the canvas size has changed, recreate the background image
	w = gui_param_w;
	h = gui_param_h;
	if (w_old != w || h_old != h)
	{
		img_screen_background = Mat(h, w, CV_8UC3, Scalar(255, 255, 255)); // white background
		w_old = w; h_old = h;
	}

	// update marker parameters
	calibration.ar_canvas.marker_size = round(gui_param_marker_size);


	//timer.tick();

	// read image data from eye- and scene camera
	eye_camera->read(frame_eye_cam);
	scene_camera->read(frame_scene_cam);

		
	// ********************************************************
	// get pupil position
	/*
	cv::cvtColor(frame_eye_cam, frame_eye_gray, cv::COLOR_BGR2GRAY);
	if (opt.blur > 0) { GaussianBlur(frame_eye_gray, frame_eye_gray, cv::Size(opt.blur, opt.blur), 0); }
	std::tie(pupil_pos, pupil_pos_coarse) = timm.pupil_center(frame_eye_gray);
	
	timm.visualize_frame(frame_eye_gray, pupil_pos, pupil_pos_coarse);
	*/

	pupil_tracker->update(frame_eye_cam);
	pupil_pos = pupil_tracker->pupil_center();

	// ********************************************************
	// map pupil position to scene camera position using calibrated 2d to 2d mapping
	p_calibrated = calibration.mapping_2d_to_2d(Point2f(pupil_pos.x, pupil_pos.y));

	// ********************************************************
	calibration.ar_canvas.update(frame_scene_cam);
	if (calibration.ar_canvas.valid())
	{
		p_projected = calibration.ar_canvas.transform(p_calibrated);
		p_projected -= calibration.offset;
	}

	// uncomment this to simulate gaze using the computer mouse
	// p_projected = Point2f(mx, my);

	// jitter filter
	p_projected.x = gaze_filter_x(p_projected.x);
	p_projected.y = gaze_filter_y(p_projected.y);


	// ********************************************************
	// event propagation and keyboard handling 
	key_pressed = cv::waitKey(1);


	// ********************************************************
	// call update functions specific to current state / task
	switch (state)
	{
	case STATE_CALIBRATION:
		calibration.update(frame_scene_cam, pupil_pos, key_pressed);
		break;
	case STATE_RUNNING:
		if (calibration.ar_canvas.valid() && int(' ') == key_pressed)
		{
			// cout << "calibrated point = " << p_calibrated.x << "\t" << p_calibrated.y << endl;
			// use space as acknowledgement for a selected letter
			eye_button_up = true;
		}
		break;
	};

	sg.update();
	eye_cam_controls.update();
}



void Eyetracking_speller::run(enum_simd_variant simd_width,  int eye_cam_id, int scene_cam_id)
{
	cv::setUseOptimized(true);
	eye_camera = select_camera("select eye camera number (0..n):", eye_cam_id);
	scene_camera = select_camera("select scene camera number (0..n):", scene_cam_id);


	cout << "\nTo improve calibration results, the autofocus of both eye- and scene camera will be disabled.\n";
	cout << "\n The autofocus can be turned on/off via the camera menu.";
	//cout << "disable autofocus of both cameras (y/n):"; char c; cin >> c;
	//if (c == 'y')
	{
		eye_camera->set(cv::CAP_PROP_AUTOFOCUS, 0);
		scene_camera->set(cv::CAP_PROP_AUTOFOCUS, 0);
		cout << "\nautofocus disabled.\n";
	}

	setup(simd_width);

	// main loop
	Timer timer(50);
	while (is_running)
	{
		timer.tick();
		// for gui stuff
		//opt = set_options(params);
		//timm.set_options(opt);

		Pupil_tracking::update();
		//Pupil_tracking::draw();
		update();
		draw();

		if (27 == key_pressed) // VK_ESCAPE
		{
			break;
		}
		timer.tock();
	}
}


#ifdef LSL_ENABLED
#pragma comment(lib, "liblsl64.lib")
#include <lsl_cpp.h>
#include "lt_lsl_protocol.h"
#include <limits>

//#include "deps/s/sdl_opencv.h"

// multithreaded capture and rendering to ensure flicker stimuli are presented with the monitor refresh rate
// separate blocking function with a while loop
void Eyetracking_speller::run_ssvep()
{
	using namespace cv;
	using namespace lsl;
	using namespace chrono;
	
	cv::destroyAllWindows();

	//hide gui to avoid multithreading problems (especially with changing camera properties )
	sg.hide();
	bool run = true;
	Simple_gui sg_local(50,50,150,100, "Record and Stream");
	sg_local.add_button("stop streaming", [&](){ run = false;  });
	//sg_local.add_button("quit program", [&]() { exit(EXIT_SUCCESS);} );
	sg_local.finish();
	
	// Sdl_opencv sdl;

	
	//////////////////////////////
	// launch the capture threads
	thread_eyecam.setup(eye_camera, "eyecam");
	thread_scenecam.setup(scene_camera, "scncam");

	cout << "waiting for the first video frames to arrive..";
	// wait for frames to arrive
	while (!(thread_eyecam.new_frame && thread_scenecam.new_frame)) { cout << "."; cv::waitKey(1); this_thread::sleep_for(150ms); }
	cout << "\nthe first frame of both the eye- and scenecam has arrived.\n";
	//////////////////////////////



	vector<double> eye_data(LT_N_EYE_DATA);
	vector<double> marker_data(1+4*2);

	// labstreaming layer 
	// todo: add correct sampling rate
	cout << "creating labstreaming layer outlet for simulated EEG data..\n";
	stream_outlet lsl_out_eye(stream_info("LT_EYE", "LT_EYE", LT_N_EYE_DATA, 30, cf_double64));
	stream_outlet lsl_out_marker(stream_info("LT_MARKER", "LT_MARKER", 1 + 4 * 2, 30, cf_double64));


	auto time_start = chrono::high_resolution_clock::now();
	
	Timer timer0(500, "\nframe :"); // man duration of individual frames. for 60 Hz monitor refresh rate, it should be close to 16.66 ms
	Timer timer1(500, "\nupdate:");
	Timer timer2(500, "\nrender:");

	while (run)
	{
		//timer0.tick();
		//timer1.tick();

		// process events
		//if (sdl.waitKey().sym == SDLK_ESCAPE) { break; }

		// ********************************************************
		// copy camera data from the capture threads
		if (thread_scenecam.new_frame)
		{
			thread_scenecam.get_frame(frame_scene_cam);
			thread_scenecam.new_frame = false;

			calibration.ar_canvas.update(frame_scene_cam);

			// send marker data (helpful in the client for visualizing marker positions
			marker_data[0] = duration_cast<duration<double>>(high_resolution_clock::now() - time_start).count();
			for (int i = 0; i < calibration.ar_canvas.image_plane.size(); i++)
			{
				marker_data[1 + 2 * i + 0] = double(calibration.ar_canvas.image_plane[i].x) / frame_scene_cam.cols;
				marker_data[1 + 2 * i + 1] = double(calibration.ar_canvas.image_plane[i].y) / frame_scene_cam.rows;
			}
			lsl_out_marker.push_sample(marker_data);
		}

		// TODO: if the eye cam has a higher FPS than the render thread, the pupil center calculation should be in a separate thread !
		if (thread_eyecam.new_frame)
		{
			thread_eyecam.get_frame(frame_eye_cam);
			thread_eyecam.new_frame = false;

			// ********************************************************
			// get pupil position
			/*
			cv::cvtColor(frame_eye_cam, frame_eye_gray, cv::COLOR_BGR2GRAY);
			if (opt.blur > 0) { GaussianBlur(frame_eye_gray, frame_eye_gray, cv::Size(opt.blur, opt.blur), 0); }
			std::tie(pupil_pos, pupil_pos_coarse) = timm.pupil_center(frame_eye_gray);
			*/
			pupil_tracker->update(frame_eye_cam);
			pupil_pos = pupil_tracker->pupil_center();

			// ********************************************************
			// map pupil position to scene camera position using calibrated 2d to 2d mapping
			p_calibrated = calibration.mapping_2d_to_2d(Point2f(pupil_pos.x, pupil_pos.y));

			//std::nan
			for (auto& x : eye_data) { x = nan(); }

			eye_data[LT_TIMESTAMP] = duration_cast<duration<double>>(high_resolution_clock::now() - time_start).count();
			eye_data[LT_PUPIL_X] = double(pupil_pos.x) / frame_eye_cam.cols;
			eye_data[LT_PUPIL_Y] = double(pupil_pos.y) / frame_eye_cam.rows;
			eye_data[LT_GAZE_X] = double(p_calibrated.x) / frame_scene_cam.cols;
			eye_data[LT_GAZE_Y] = double(p_calibrated.y) / frame_scene_cam.rows;

			if (calibration.ar_canvas.valid())
			{
				p_projected = calibration.ar_canvas.transform(p_calibrated, calibration.ar_canvas.screen_plane_external);
				p_projected -= calibration.offset;

				// jitter filter (updated at eyecam fps)
				eye_data[LT_SCREEN_X] = p_projected.x;
				eye_data[LT_SCREEN_Y] = p_projected.y;
				eye_data[LT_SCREEN_X_FILTERED] = gaze_filter_x(p_projected.x);
				eye_data[LT_SCREEN_Y_FILTERED] = gaze_filter_y(p_projected.y);
			}

			lsl_out_eye.push_sample(eye_data);
		}

		
		/*
		// uncomment this to simulate gaze using the computer mouse
		//p_projected = Point2f(mx, my);

		timer1.tock();

		//* // old code for vertically synchronized rendering using libSDL 
		// render part
		timer2.tick();
		img_screen_background.copyTo(img_screen);
		draw_speller(true);
		timer2.tock();

		
		
		// draw to screen (vsynced flip) 
		sdl.imshow(img_screen,100,100);
		auto dt = timer0.tock();
		if (dt > 0.025)
		{
			cout << "\nslow frame. dt = " << dt;
		}
		*/

		sg_local.update();
	}


	thread_scenecam.stop();
	thread_eyecam.stop();
	sg.show();

	// todo restore all other windows 

}

#else
void Eyetracking_speller::run_ssvep()
{
}
#endif
