#ifndef TIMING_H
#define TIMING_H

#if defined(_WIN32)

#define CLOCK_REALTIME_WIN32 0
#include <windows.h>

struct timespec_win32
{
	__int64 tv_sec;
	__int64 tv_nsec;
};

#define exp7           10000000LL     //1E+7     //C-file part
#define exp9         1000000000LL     //1E+9
#define w2ux 116444736000000000LL     //1.jan1601 to 1.jan1970

static void unix_time(struct timespec_win32 *spec)
{
	__int64 wintime;
	GetSystemTimeAsFileTime((FILETIME*)&wintime); 
	wintime -= w2ux;
	spec->tv_sec  = wintime / exp7;                 
	spec->tv_nsec = wintime % exp7 *100;
}

static int clock_gettime_win32(int, timespec_win32 *spec)
{
	static struct timespec_win32 startspec;
	static double ticks2nano;
	static __int64 startticks, tps = 0;
	__int64 tmp, curticks;

	// A system can possibly change the freq.
	QueryPerformanceFrequency((LARGE_INTEGER*)&tmp);
	if (tps != tmp)
	{
		tps = tmp;
		QueryPerformanceCounter((LARGE_INTEGER*)&startticks);
		unix_time(&startspec);
		ticks2nano = (double)exp9 / tps;
	}

	QueryPerformanceCounter((LARGE_INTEGER*)&curticks); curticks -=startticks;
	spec->tv_sec  =startspec.tv_sec + (curticks / tps);
	spec->tv_nsec =startspec.tv_nsec + (double)(curticks % tps) * ticks2nano;

	if (!(spec->tv_nsec < exp9))
	{
		spec->tv_sec++;
		spec->tv_nsec -=exp9;
	}

	return 0;
}

#else

#include <time.h>

#endif // defined(_WIN32) && !defined(__MINGW32__)

// Get the timer value.
static void get_time(double* ret)
{
#ifdef _WIN32
	volatile struct timespec val;
	clock_gettime_win32(CLOCK_REALTIME_WIN32, (struct timespec_win32*)&val);
#else
	volatile struct timespec val;
	clock_gettime(CLOCK_REALTIME, (struct timespec*)&val);
#endif
	*ret = (double)0.000000001 * val.tv_nsec + val.tv_sec;
}

#endif // TIMING_H

