#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <cmath>
// simple, matlab like timing measurements.
// example:
// Timer timer(10);
// while(1)
// {
//		timer.tick();
//		some_time_consuming_operation();
//		timer.tock();
// }
class Timer
{
protected:
	std::chrono::duration<double> duration_accum = std::chrono::duration<double>(0);
	double sd_accum = 0.0f; // acuum for standard deviation 
	double mean_duration = 0.0f;
	int counter = 0;
	int n_measurements = 1;
	std::chrono::high_resolution_clock::time_point t1;
	std::chrono::high_resolution_clock::time_point t_launched;
	std::string message;
public:
	
	Timer(int n_measurements_ = 1, std::string message_ = "\nmean duration = ")
	{
		message = message_;
		n_measurements = n_measurements_;
		t_launched = std::chrono::high_resolution_clock::now();
	}


	// returns time in s
	double get_time()
	{
		using namespace std::chrono;
		return  duration_cast<duration<double>>(high_resolution_clock::now() - t_launched).count();
	}

	void tick()
	{
		t1 = std::chrono::high_resolution_clock::now();
	}


	// returns delta t between tick and tock. 
	// prints the mean duration in s after n_measurements
	double tock(bool print_result = true)
	{
		using namespace std::chrono;
		auto dt = duration_cast<duration<double>>(high_resolution_clock::now() - t1);

		// calc mean and SD over n_repetitions
		if (mean_duration != 0.0f) { double tmp = mean_duration - dt.count();  sd_accum += tmp*tmp; }
		duration_accum += dt;
		counter++;
		if (counter == n_measurements)
		{
			mean_duration = duration_accum.count() / double(n_measurements);
			
			if (print_result)
			{ 
				std::cout << message << 1000.0f * mean_duration << " ms";
			
				// calculating the standard deviation without enough samples doesnt really make any sense
				if(n_measurements >=5)
				{
					double SD = std::sqrt(sd_accum / double(n_measurements - 1.0));
					std::cout << "\t" << "SD = " << 1000.0f * SD << " ms";
				}
			}

			duration_accum = duration<double>(0);
			sd_accum = 0;
			counter = 0;
		}

		return dt.count();
	}
};


#ifdef __TEST_THIS_MODULE__

#include<cstdlib>

auto some_time_consuming_operation()
{
	return sqrt(cos(tanh(sin(log(sqrt(100.0f * rand() / RAND_MAX))))));
}

void test_module()
{
	Timer timer(100);
	
	for(int i=0;i<1000;i++)
	{
		timer.tick();
		float a = 0.0f;
		for (int k = 0; k < 1000; k++)
		{
			a += some_time_consuming_operation();
		}
		timer.tock();
	}
}

#endif