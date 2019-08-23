#pragma once

#include <string>

#include <opencv2/opencv.hpp>
#include <SDL2/SDL.h>

// helper class to draw an OpenCV Mat with vertical sync enabled 
// mostly simulates the equivalent OpenCV interface 
class Sdl_opencv 
{
protected:
	unsigned int w = 0;
	unsigned int h = 0;
	std::string title_ = "";
	SDL_Window* window=nullptr;
	SDL_Renderer* renderer = nullptr;
	SDL_Surface* surface = nullptr;
	SDL_Texture* texture = nullptr;
	bool initialized = false;
	void re_create_surface(cv::Mat& image);
public:
	Sdl_opencv() {}
	Sdl_opencv(std::string title) { title_ = title; }
	~Sdl_opencv();

	// shows an opencv mat and wait for vertical sync
	void imshow(cv::Mat& image, int x, int y);
	SDL_Keysym waitKey();
};

#ifdef __TEST_THIS_MODULE__
void test_module_sdl_opencv();
#endif