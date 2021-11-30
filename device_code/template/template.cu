#include "template.h"

AllocatorHandle<AllocatorT>* allocator_handle;
__device__ AllocatorT* device_allocator;

__device__ SampleClass::SampleClass(int seed) {
	curand_init(kSeed, seed, 0, &random_state_);
	this->x = curand(&random_state_) % 50;
	this->y = curand(&random_state_) % 50;
}

__device__ void SampleClass::update() {
	int rand1 = (curand(&random_state_) % 3) - 1;
	int rand2 = (curand(&random_state_) % 3) - 1;
	this->x += rand1;
	this->y += rand2;
}

void SampleClass::_do(void (*pf)(int, int, int)){
	pf(0, this->x, this->y);
}

extern "C" int SampleClass_do_all(void (*pf)(int, int, int)){
	allocator_handle->template device_do<SampleClass>(&SampleClass::_do, pf);
 	return 0;
}

extern "C" int SampleClass_SampleClass_update(){
	allocator_handle->parallel_do<SampleClass, &SampleClass::update>();
	return 0;
}

extern "C" int parallel_new_SampleClass(int object_num){
	allocator_handle->parallel_new<SampleClass>(object_num);
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