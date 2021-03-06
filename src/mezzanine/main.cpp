#include "pupil_tracker.h"
#include "opencv2/imgcodecs.hpp"
#include "timing.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <experimental/filesystem>
#include <fenv.h>
#include <memory>
#include <vector>

using namespace std;

namespace fs = std::experimental::filesystem;

int main(int argc, char* argv[])
{
	static const fs::path directory = "data/images/trial_0/raw";

	unique_ptr<PupilTracker> pupilTracker;
	
	// Queue images in the target folder.
	if (fs::exists(directory) && fs::is_directory(directory))
	{
		// Get the list of images.
		std::vector<string> imageFilenames;
		for (fs::directory_iterator i(directory), ie; i != ie; i++)
		{
			if (!fs::is_regular_file(i->status())) continue;			

			if (i->path().extension() == ".png")
				imageFilenames.push_back(i->path().filename().string());
		};

		// Sort images before processing.
		sort(imageFilenames.begin(), imageFilenames.end());

		// Process images one by one.
		double avgDuration = 0.0;
		fs::path imagePath(directory);
		fs::create_directory(imagePath.parent_path() / "processed");
		for (int i = 0, ie = imageFilenames.size(); i < ie; i++)
		{
			const string inputPath = (imagePath / imageFilenames[i]).string();
			cv::Mat image = cv::imread(inputPath, cv::IMREAD_GRAYSCALE);

			if (!pupilTracker)
				pupilTracker.reset(new PupilTracker());

			cv::Point pupil_pos, pupil_pos_coarse;
			double start; get_time(&start);
			pupilTracker->run(image, pupil_pos, pupil_pos_coarse);
			double finish; get_time(&finish);
			pupilTracker->annotate(image, pupil_pos, pupil_pos_coarse);

			const string outputPath = (imagePath.parent_path() / "processed" / imageFilenames[i]).string();
			cv::imwrite(outputPath, image);

			const double time_ms = (finish - start) * 1000;
			printf("Processing frame \"%s\" time = %f ms\n", imageFilenames[i].c_str(), time_ms);
			if (i > 0) avgDuration += time_ms;
		}
		
		printf("Average processing time = %f ms\n", avgDuration / (imageFilenames.size() - 1));
	}

	return 0;
}

