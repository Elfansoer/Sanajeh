#ifndef SANAJEH_DEVICE_CODE_H
#define SANAJEH_DEVICE_CODE_H
#define KNUMOBJECTS 64*64*64*64

#include <curand_kernel.h>
#include "dynasoar.h"

class Body;


using AllocatorT = SoaAllocator<KNUMOBJECTS, Body>;

static const int kSeed = 45;
static const float kMaxMass = 1000.0;
static const float kDt = 0.01;
static const float kGravityConstant = 4e-06;
static const float kDampeningFactor = 0.05;

class Body : public AllocatorT::Base {
	public:
		declare_field_types(Body, curandState, float, float, float, float, float, float, float)
		Field<Body, 0> random_state_;
		Field<Body, 1> pos_x;
		Field<Body, 2> pos_y;
		Field<Body, 3> vel_x;
		Field<Body, 4> vel_y;
		Field<Body, 5> force_x;
		Field<Body, 6> force_y;
		Field<Body, 7> mass;
	
		__device__ Body(int idx);
		__device__ void compute_force();
		__device__ void apply_force(Body* other);
		__device__ void update();
		void _do(void (*pf)(int, float, float, float, float, float, float, float));
};

extern "C" int Body_do_all(void (*pf)(int, float, float, float, float, float, float, float));
extern "C" int Body_Body_compute_force();
extern "C" int Body_Body_update();
extern "C" int parallel_new_Body(int object_num);
extern "C" int AllocatorInitialize();

extern "C" int AllocatorUninitialize();

#endif