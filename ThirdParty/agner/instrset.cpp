// AVX instruction set detection
// Based on Agner Fog's C++ vector class library:
// http://www.agner.org/optimize/vectorclass.zip
#include "instrset.h"

#include <stdint.h>
#include <x86intrin.h>

using namespace agner;

// Define interface to cpuid instruction.
// input:  eax = functionnumber, ecx = 0
// output: eax = output[0], ebx = output[1], ecx = output[2], edx = output[3]
static inline void cpuid (int output[4], int functionnumber)
{	
	int a, b, c, d;
	__asm("cpuid" : "=a"(a),"=b"(b),"=c"(c),"=d"(d) : "a"(functionnumber),"c"(0) : );
	output[0] = a;
	output[1] = b;
	output[2] = c;
	output[3] = d;
}

// Define interface to xgetbv instruction
static inline int64_t xgetbv (int ctr)
{
	uint32_t a, d;
	__asm("xgetbv" : "=a"(a),"=d"(d) : "c"(ctr) : );
	return a | (((uint64_t) d) << 32);
}

InstrSet agner::InstrSetDetect()
{
	static InstrSet iset = InstrSetUnknown;

	if (iset != InstrSetUnknown) return iset;

	iset = InstrSet80386;                                  // default value
	int abcd[4] = { 0, 0, 0, 0 };                          // cpuid results
	cpuid(abcd, 0);                                        // call cpuid function 0
	if (abcd[0] == 0) return iset;                         // no further cpuid function supported
	cpuid(abcd, 1);                                        // call cpuid function 1 for feature flags
	if ((abcd[3] & (1 <<  0)) == 0) return iset;           // no floating point
	if ((abcd[3] & (1 << 23)) == 0) return iset;           // no MMX
	if ((abcd[3] & (1 << 15)) == 0) return iset;           // no conditional move
	if ((abcd[3] & (1 << 24)) == 0) return iset;           // no FXSAVE
	if ((abcd[3] & (1 << 25)) == 0) return iset;           // no SSE
	iset = InstrSetSSE;                                    // 1: SSE supported
	if ((abcd[3] & (1 << 26)) == 0) return iset;           // no SSE2
	iset = InstrSetSSE2;                                   // 2: SSE2 supported
	if ((abcd[2] & (1 <<  0)) == 0) return iset;           // no SSE3
	iset = InstrSetSSE3;                                   // 3: SSE3 supported
	if ((abcd[2] & (1 <<  9)) == 0) return iset;           // no SSSE3
	iset = InstrSetSSSE3;                                  // 4: SSSE3 supported
	if ((abcd[2] & (1 << 19)) == 0) return iset;           // no SSE4.1
	iset = InstrSetSSE41;                                  // 5: SSE4.1 supported
	if ((abcd[2] & (1 << 23)) == 0) return iset;           // no POPCNT
	if ((abcd[2] & (1 << 20)) == 0) return iset;           // no SSE4.2
	iset = InstrSetSSE42;                                  // 6: SSE4.2 supported
	if ((abcd[2] & (1 << 27)) == 0) return iset;           // no OSXSAVE
	if ((xgetbv(0) & 6) != 6)       return iset;           // AVX not enabled in O.S.
	if ((abcd[2] & (1 << 28)) == 0) return iset;           // no AVX
	iset = InstrSetAVX;                                    // 7: AVX supported
	cpuid(abcd, 7);                                        // call cpuid leaf 7 for feature flags
	if ((abcd[1] & (1 <<  5)) == 0) return iset;           // no AVX2
	iset = InstrSetAVX2;                                   // 8: AVX2 supported
	if ((abcd[1] & (1 << 16)) == 0) return iset;           // no AVX512
	cpuid(abcd, 0xD);                                      // call cpuid leaf 0xD for feature flags
	if ((abcd[0] & 0x60) != 0x60)   return iset;           // no AVX512
	iset = InstrSetAVX512F;                                // 9: AVX512F supported
	cpuid(abcd, 7);                                        // call cpuid leaf 7 for feature flags
	if ((abcd[1] & (1 << 31)) == 0) return iset;           // no AVX512VL
	iset = InstrSetAVX512VL;                               // 10: AVX512VL supported
	if ((abcd[1] & 0x40020000) != 0x40020000) return iset; // no AVX512BW, AVX512DQ
	iset = InstrSetAVX512BW_DQ;                            // 11: AVX512BW, AVX512DQ supported

	return iset;
}

bool agner::isSupported(InstrSet iset)
{
	if (iset == InstrSetUnknown) return false;

	InstrSet isetMax = InstrSetDetect();
	return (iset <= isetMax);
}

