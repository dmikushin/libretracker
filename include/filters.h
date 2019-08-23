#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <array>
#include <exception>
#include <vector>


// double moving average filter
// see skeletal joint smoothing white paper
template<typename T> class Filter_double_ma
{
protected:
	T y = 0;
	T x1 = 0; // 1 timestep older
	T x2 = 0; // 2 timesteps older
	T x3 = 0; // ...
	T x4 = 0; 
	T x5 = 0;
	T w1 =   5.0 / 9.0;
	T w2 =   4.0 / 9.0;
	T w3 =   1.0 / 3.0;
	T w4 = - 2.0 / 9.0;
	T w5 = - 1.0 / 9.0;
public:
	const T& operator()()            { return y; }
	const T& operator()(const T& x0) { return update(x0); }
	const T& get_last() { return y; }	
	const T& update(const T& x0)
	{
		y = w1 * x0 + w2 * x1 + w3 * x2 + w4 * x3 + w5 * x5;
		// slide window 
		x5 = x4; x4 = x3; x3 = x2; x2 = x1; x1 = x0;
		return y;
	}

	void reset(const T& x0 = 0) { y = x1 = x2 = x3 = x4 = x0; }
};

template<typename T> class Filter_lowpass
{
protected:
	T y = 0;
	T alpha = 0.8;
public:
	Filter_lowpass(const T& alpha_) { alpha = alpha_; }
	const T& operator()() { return y; }
	const T& operator()(const T& x0) { return update(x0); }
	const T& get_last() { return y; }
	const T& update(const T& x0)
	{
		y = alpha*x0 + (1 - alpha)*y;
		return y;
	}
	void reset(const T& x0 = 0) { y = x0; }
};

template<typename T> class Filter_double_exponential
{
protected:
	T y0 = 0;
	T y1 = 0; // 1 timestep older
	T alpha = 0.25f; // larger  values: less smoothing
	T gamma = 0.075f; // smaller values: less overshoot, less lag reduction
	T v = 0; // corresponds to the smoothed joint velocity

public:
	//Filter_double_exponential(const T& alpha_=0.05, const T& gamma_=0.1) { alpha = alpha_; gamma = gamma_; }
	const T& operator()() { return y0; }
	const T& operator()(const T& x0) { return update(x0); }
	const T& get_last() { return y0; }
	const T& update(const T& x0)
	{		
		y0 = alpha*x0 + (1 - alpha)*(y1 + v);
		v = gamma*(y0 - y1) + (1 - gamma)*v;
		y1 = y0;
		return y0;
	}
	const T& forecast(unsigned int k = 1) { return y0 + k*v; }
	void reset(const T& x0 = 0) { y0 = y1 = x0; v = 0; }

	void set_params(const T& alpha_, const T& gamma_) { alpha = alpha_; gamma = gamma_; }
};



/////////////////////////////////////////////////////////////////////////////////

// 2016: butterworth modified, after
// http://www.exstrom.com/journal/sigproc/

/*
*                            COPYRIGHT
*
*  Copyright (C) 2014 Exstrom Laboratories LLC
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  A copy of the GNU General Public License is available on the internet at:
*  http://www.gnu.org/copyleft/gpl.html
*
*  or you can write to:
*
*  The Free Software Foundation, Inc.
*  675 Mass Ave
*  Cambridge, MA 02139, USA
*
*  Exstrom Laboratories LLC contact:
*  stefan(AT)exstrom.com
*
*  Exstrom Laboratories LLC
*  Longmont, CO 80503, USA
*
*/




// The order of the filter must be a multiple of 4.
// n = filter order 4,8,12,... 
template<class T, int n> class Butter_bandpass
{
protected:
	T s, f1, f2, a, a2, b, b2, r;
	// todo make constexpr n/4
	//array<T, n> A;
	//array<T, n> d1, d2, d3, d4;
	//array<T, n> w0, w1, w2, w3, w4;

	std::vector<T> A;
	std::vector<T> d1, d2, d3, d4;
	std::vector<T> w0, w1, w2, w3, w4;
public:

	// s = sampling frequency
	// f1 = upper half power frequency (-3 dB = 0.5 * power or 1/sqrt(2) = 0.707 * gain ; equivalent to the most common definition of "cut-off" / "edge" / "corner" frequency)
	// f2 = lower half power frequency
	Butter_bandpass(float s_, float f1_, float f2_)
	{

		if (n % 4)
		{
			const char* err_msg = "Order must be 4,8,12,16,...";
			//std::cerr << err_msg << std::endl;			
			//std::throw(std::exception(err_msg));
		}

		s = s_;
		f1 = f1_;
		f2 = f2_;

		init();
	}

	void reset() { init(); }

	void init()
	{
		a = cos(M_PI*(f1 + f2) / s) / cos(M_PI*(f1 - f2) / s);
		a2 = a*a;
		b = tan(M_PI*(f1 - f2) / s);
		b2 = b*b;


		int n2 = n / 4;
		A.resize(n2);
		d1.resize(n2);
		d2.resize(n2);
		d3.resize(n2);
		d4.resize(n2);

		// these need to be set to zero
		w0.resize(n2);
		w1.resize(n2);
		w2.resize(n2);
		w3.resize(n2);
		w4.resize(n2);
		for (int i = 0; i < n2; i++)
		{
			w0[i] = 0;
			w1[i] = 0;
			w2[i] = 0;
			w3[i] = 0;
			w4[i] = 0;
		}


		for (int i = 0; i<n2; ++i)
		{
			r = sin(M_PI*(2.0*i + 1.0) / (4.0*n));
			s = b2 + 2.0*b*r + 1.0;
			A[i] = b2 / s;
			d1[i] = 4.0*a*(1.0 + b*r) / s;
			d2[i] = 2.0*(b2 - 2.0*a2 - 1.0) / s;
			d3[i] = 4.0*a*(1.0 - b*r) / s;
			d4[i] = -(b2 - 2.0*b*r + 1.0) / s;
		}
	}

	T update(T x)
	{
		int n2 = n / 4;
		for (int i = 0; i<n2; ++i)
		{
			w0[i] = d1[i] * w1[i] + d2[i] * w2[i] + d3[i] * w3[i] + d4[i] * w4[i] + x;
			x = A[i] * (w0[i] - 2.0*w2[i] + w4[i]);
			w4[i] = w3[i];
			w3[i] = w2[i];
			w2[i] = w1[i];
			w1[i] = w0[i];
		}
		return x;
	}
};

/////////////////////////////////////////////////////////////////////////////////