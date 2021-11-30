#ifndef SANAJEH_DEVICE_CODE_H
#define SANAJEH_DEVICE_CODE_H
#define KNUMOBJECTS 64*64*64*64

#include <curand_kernel.h>
#include "dynasoar.h"

class Cell;


using AllocatorT = SoaAllocator<KNUMOBJECTS, Cell>;

static const int kSeed = 43;
static const int kSizeX = 100;
static const int kSizeY = 100;

class Cell : public AllocatorT::Base {
	public:
		declare_field_types(Cell, curandState, int, int, int, int, int, int, int)
		Field<Cell, 0> random_state_;
		Field<Cell, 1> x;
		Field<Cell, 2> y;
		Field<Cell, 3> r;
		Field<Cell, 4> g;
		Field<Cell, 5> b;
		Field<Cell, 6> life;
		Field<Cell, 7> birth;
	
		__device__ Cell(int seed);
		__device__ void update();
		void _do(void (*pf)(int, int, int, int, int, int, int, int));
};

extern "C" int Cell_do_all(void (*pf)(int, int, int, int, int, int, int, int));
extern "C" int Cell_Cell_update();
extern "C" int parallel_new_Cell(int object_num);
extern "C" int AllocatorInitialize();

extern "C" int AllocatorUninitialize();

#endif