#ifndef SANAJEH_DEVICE_CODE_H
#define SANAJEH_DEVICE_CODE_H
#define KNUMOBJECTS 64*64*64*64

#include <curand_kernel.h>
#include "dynasoar.h"

class SampleClass;


using AllocatorT = SoaAllocator<KNUMOBJECTS, SampleClass>;

static const int kSeed = 43;

class SampleClass : public AllocatorT::Base {
	public:
		declare_field_types(SampleClass, curandState, int, int)
		Field<SampleClass, 0> random_state_;
		Field<SampleClass, 1> x;
		Field<SampleClass, 2> y;
	
		__device__ SampleClass(int seed);
		__device__ void update();
		void _do(void (*pf)(int, int, int));
};

extern "C" int SampleClass_do_all(void (*pf)(int, int, int));
extern "C" int SampleClass_SampleClass_update();
extern "C" int parallel_new_SampleClass(int object_num);
extern "C" int AllocatorInitialize();

extern "C" int AllocatorUninitialize();

#endif