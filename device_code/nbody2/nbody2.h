#ifndef SANAJEH_DEVICE_CODE_H
#define SANAJEH_DEVICE_CODE_H
#define KNUMOBJECTS 64*64*64*64

#include <curand_kernel.h>
#include "dynasoar.h"

class Projectile;


using AllocatorT = SoaAllocator<KNUMOBJECTS, Projectile>;


class Projectile : public AllocatorT::Base {
	public:
		declare_field_types(Projectile, float, float)
		Field<Projectile, 0> x;
		Field<Projectile, 1> y;
	
		__device__ Projectile(int idx);
		__device__ void update();
		void _do(void (*pf)(float, float));
};

extern "C" int Projectile_do_all(void (*pf)(float, float));
extern "C" int Projectile_Projectile_update();
extern "C" int parallel_new_Projectile(int object_num);
extern "C" int AllocatorInitialize();

extern "C" int AllocatorUninitialize();

#endif