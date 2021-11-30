#include "nbody2.h"

AllocatorHandle<AllocatorT>* allocator_handle;
__device__ AllocatorT* device_allocator;

__device__ Projectile::Projectile(int idx) {

}

__device__ void Projectile::update() {
	this->x += 10;
	this->y += 10;
}

void Projectile::_do(void (*pf)(float, float)){
	pf(this->x, this->y);
}

extern "C" int Projectile_do_all(void (*pf)(float, float)){
	allocator_handle->template device_do<Projectile>(&Projectile::_do, pf);
 	return 0;
}

extern "C" int Projectile_Projectile_update(){
	allocator_handle->parallel_do<Projectile, &Projectile::update>();
	return 0;
}

extern "C" int parallel_new_Projectile(int object_num){
	allocator_handle->parallel_new<Projectile>(object_num);
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