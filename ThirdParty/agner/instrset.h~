#ifndef INSTRSET_H
#define INSTRSET_H

enum InstrSet
{
	InstrSetUnknown     = -1,
	InstrSet80386       =  0, // 80386 instruction set
	InstrSetSSE         =  1, // or above = SSE (XMM) supported by CPU (not testing for O.S. support)
	InstrSetSSE2        =  2, // or above = SSE2
	InstrSetSSE3        =  3, // or above = SSE3
	InstrSetSSSE3       =  4, // or above = Supplementary SSE3 (SSSE3)
	InstrSetSSE41       =  5, // or above = SSE4.1
	InstrSetSSE42       =  6, // or above = SSE4.2
	InstrSetAVX         =  7, // or above = AVX supported by CPU and operating system
	InstrSetAVX2        =  8, // or above = AVX2
	InstrSetAVX512F     =  9, // or above = AVX512F
	InstrSetAVX512VL    = 10, // or above = AVX512VL
	InstrSetAVX512BW_DQ = 11, // or above = AVX512BW, AVX512DQ
};

InstrSet InstrSetDetect();

bool isSupported(InstrSet iset);

#endif // INSTRSET_H

