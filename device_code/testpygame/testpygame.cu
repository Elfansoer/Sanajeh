#include "testpygame.h"

AllocatorHandle<AllocatorT>* allocator_handle;
__device__ AllocatorT* device_allocator;

__device__ Cell::Cell(int seed) {
	curand_init(kSeed, seed, 0, &random_state_);
	this->x = curand(&random_state_) % kSizeX;
	this->y = curand(&random_state_) % kSizeY;
	this->r = curand(&random_state_) % 255;
	this->g = curand(&random_state_) % 255;
	this->b = curand(&random_state_) % 255;
	this->life = 20;
	this->birth = 10;
}

__device__ void Cell::update() {
	int rand = curand(&random_state_) % 4;
	if (rand == 0) {
		this->x = (this->x + 1) % kSizeX;
	}
	if (rand == 1) {
		this->x = (this->x - 1) % kSizeX;
	}
	if (rand == 2) {
		this->y = (this->y + 1) % kSizeY;
	}
	if (rand == 3) {
		this->y = (this->y + 1) % kSizeY;
	}
	this->birth -= 1;
	this->life -= 1;
	if (this->birth == 0) {
		this->birth = 10;
		Cell* new_cell = new(device_allocator) Cell(curand(&random_state_));
		new_cell->x = this->x;
		new_cell->y = this->y;
	}
	if (this->life == 0) {
		destroy(device_allocator, this);
	}
}

void Cell::_do(void (*pf)(int, int, int, int, int, int, int, int)){
	pf(0, this->x, this->y, this->r, this->g, this->b, this->life, this->birth);
}

extern "C" int Cell_do_all(void (*pf)(int, int, int, int, int, int, int, int)){
	allocator_handle->template device_do<Cell>(&Cell::_do, pf);
 	return 0;
}

extern "C" int Cell_Cell_update(){
	allocator_handle->parallel_do<Cell, &Cell::update>();
	return 0;
}

extern "C" int parallel_new_Cell(int object_num){
	allocator_handle->parallel_new<Cell>(object_num);
	return 0;
}

extern "C" int AllocatorInitialize(){
	allocator_handle = new AllocatorHandle<AllocatorT>(/* unified_memory= */ true);
	AllocatorT* dev_ptr = allocator_handle->device_pointer();
	cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0, cudaMemcpyHostToDevice);
	return 0;
}

extern "C" int AllocatorUninitialize(){
	return 0;
}