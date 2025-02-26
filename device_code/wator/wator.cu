#include "wator.h"

AllocatorHandle<AllocatorT>* allocator_handle;
__device__ AllocatorT* device_allocator;
__device__ Cell* cells[kSizeX * kSizeY];

__device__ Cell::Cell(int cell_id) {
	curand_init(kSeed, cell_id, 0, &random_state_);
	this->agent_ref = nullptr;
	this->id_ = cell_id;
	this->prepare();
	cells[cell_id] = this;
}

__device__ void Cell::setup() {
	int x = this->id_ % kSizeX;
	int y = this->id_ / kSizeX;
	Cell* left = ((x > 0) ? (cells[((y * kSizeX) + x) - 1]) : (cells[((y * kSizeX) + kSizeX) - 1]));
	Cell* right = ((x < kSizeX - 1) ? (cells[((y * kSizeX) + x) + 1]) : (cells[y * kSizeX]));
	Cell* top = ((y > 0) ? (cells[((y - 1) * kSizeX) + x]) : (cells[((kSizeY - 1) * kSizeX) + x]));
	Cell* bottom = ((y < kSizeY - 1) ? (cells[((y + 1) * kSizeX) + x]) : (cells[x]));
	cells[this->id_]->set_neighbors(left, top, right, bottom);
	int agent_type = curand(&random_state_) % 4;
	if (agent_type == 0) {
		Agent* agent = new(device_allocator) Fish(curand(&random_state_));
		assert(agent != nullptr);
		cells[this->id_]->enter(agent);
	} else 	if (agent_type == 1) {
		Agent* agent = new(device_allocator) Shark(curand(&random_state_));
		assert(agent != nullptr);
		cells[this->id_]->enter(agent);
	} else {
	
	}
}

__device__ Agent* Cell::agent() {
	return this->agent_ref;
}

__device__ void Cell::decide() {
	if (this->neighbor_request_[4]) {
		this->agent_ref->set_new_position(this);
	} else {
		int candidates[4];
		int num_candidates = 0;
		for (int i = 0; i < 4; ++i) {
			if (this->neighbor_request_[i]) {
				candidates[num_candidates] = i;
				num_candidates += 1;
			}
		}
		if (num_candidates > 0) {
			int selected_index = curand(&random_state_) % num_candidates;
			Agent* __auto_v0 = this->neighbors_[candidates[selected_index]]->agent();
			__auto_v0->set_new_position(this);
		}
	}
}

__device__ void Cell::enter(Agent* agent) {
	assert(this->agent_ref == nullptr);
	assert(agent != nullptr);
	this->agent_ref = agent;
	agent->set_position(this);
}

__device__ bool Cell::has_fish() {
	return this->agent_ref->cast<Fish>() != nullptr;
}

__device__ bool Cell::has_shark() {
	return this->agent_ref->cast<Shark>() != nullptr;
}

__device__ bool Cell::is_free() {
	return this->agent_ref == nullptr;
}

__device__ void Cell::kill() {
	assert(this->agent_ref != nullptr);
	destroy(device_allocator, this->agent_ref);
	this->agent_ref = nullptr;
}

__device__ void Cell::leave() {
	assert(this->agent_ref != nullptr);
	this->agent_ref = nullptr;
}

__device__ void Cell::prepare() {
	int i = 0;
	while (i < 5) {
		this->neighbor_request_[i] = false;
		i += 1;
	}
}

__device__ void Cell::set_neighbors(Cell* left, Cell* top, Cell* right, Cell* bottom) {
	this->neighbors_[0] = left;
	this->neighbors_[1] = top;
	this->neighbors_[2] = right;
	this->neighbors_[3] = bottom;
}

__device__ void Cell::request_random_fish_neighbor() {
	if (!this->request_random_neighbor_has_fish(this->agent_ref)) {
		if (!this->request_random_neighbor_is_free(this->agent_ref)) {
			this->neighbor_request_[4] = true;
		}
	}
}

__device__ void Cell::request_random_free_neighbor() {
	if (!this->request_random_neighbor_is_free(this->agent_ref)) {
		this->neighbor_request_[4] = true;
	}
}

__device__ bool Cell::request_random_neighbor_has_fish(Agent* agent) {
	int candidates[4];
	int num_candidates = 0;
	int i = 0;
	while (i < 4) {
		if (this->neighbors_[i]->has_fish()) {
			candidates[num_candidates] = i;
			num_candidates += 1;
		}
		i += 1;
	}
	if (num_candidates == 0) {
		return false;
	} else {
		int selected_index = curand(&random_state_) % num_candidates;
		int selected = candidates[selected_index];
		int neighbor_index = (selected + 2) % 4;
		this->neighbors_[selected]->neighbor_request_[neighbor_index] = true;
		assert(this->neighbors_[selected]->neighbors_[neighbor_index] == this);
		return true;
	}
}

__device__ bool Cell::request_random_neighbor_is_free(Agent* agent) {
	int candidates[4];
	int num_candidates = 0;
	int i = 0;
	while (i < 4) {
		if (this->neighbors_[i]->is_free()) {
			candidates[num_candidates] = i;
			num_candidates += 1;
		}
		i += 1;
	}
	if (num_candidates == 0) {
		return false;
	} else {
		int selected_index = curand(&random_state_) % num_candidates;
		int selected = candidates[selected_index];
		int neighbor_index = (selected + 2) % 4;
		this->neighbors_[selected]->neighbor_request_[neighbor_index] = true;
		assert(this->neighbors_[selected]->neighbors_[neighbor_index] == this);
		return true;
	}
}

__device__ Agent::Agent(int seed) {
	curand_init(kSeed, seed, 0, &random_state_);
}

__device__ Cell* Agent::position() {
	return this->position_ref;
}

__device__ void Agent::set_new_position(Cell* new_pos) {
	assert(this->new_position_ref == this->position_ref);
	this->new_position_ref = new_pos;
}

__device__ void Agent::set_position(Cell* cell) {
	this->position_ref = cell;
}

__device__ Fish::Fish(int seed) : Agent(seed) {
	this->egg_timer_ = seed % kSpawnThreshold;
}

__device__ void Fish::prepare() {
	this->egg_timer_ += 1;
	this->new_position_ref = this->position_ref;
	assert(this->position_ref != nullptr);
	this->position_ref->request_random_free_neighbor();
}

__device__ void Fish::update() {
	Cell* old_position = this->position_ref;
	if (old_position != this->new_position_ref) {
		old_position->leave();
		this->new_position_ref->enter(this);
		if (kOptionFishSpawn && this->egg_timer_ > kSpawnThreshold) {
			Fish* new_fish = new(device_allocator) Fish(curand(&random_state_));
			assert(new_fish != nullptr);
			old_position->enter(new_fish);
			this->egg_timer_ = 0;
		}
	}
}

__device__ Shark::Shark(int seed) : Agent(seed) {
	this->energy_ = kEnergyStart;
	this->egg_timer_ = seed % kSpawnThreshold;
}

__device__ void Shark::prepare() {
	this->egg_timer_ += 1;
	this->energy_ -= 1;
	assert(this->position_ref != nullptr);
	if (kOptionSharkDie && this->energy_ == 0) {

	} else {
		this->new_position_ref = this->position_ref;
		this->position_ref->request_random_fish_neighbor();
	}
}

__device__ void Shark::update() {
	if (kOptionSharkDie && this->energy_ == 0) {
		this->position_ref->kill();
	} else {
		Cell* old_position = this->position_ref;
		if (old_position != this->new_position_ref) {
			if (this->new_position_ref->has_fish()) {
				this->energy_ += kEnergyBoost;
				this->new_position_ref->kill();
			}
			old_position->leave();
			this->new_position_ref->enter(this);
			if (kOptionSharkSpawn && this->egg_timer_ > kSpawnThreshold) {
				Shark* new_shark = new(device_allocator) Shark(curand(&random_state_));
				assert(new_shark != nullptr);
				old_position->enter(new_shark);
				this->egg_timer_ = 0;
			}
		}
	}
}

void Cell::_do(void (*pf)(int, int, int, int, int)){
	pf(0, 0, 0, this->id_, 0);
}

extern "C" int Cell_do_all(void (*pf)(int, int, int, int, int)){
	allocator_handle->template device_do<Cell>(&Cell::_do, pf);
 	return 0;
}

extern "C" int Cell_Cell_setup(){
	allocator_handle->parallel_do<Cell, &Cell::setup>();
	return 0;
}

extern "C" int Cell_Cell_prepare(){
	allocator_handle->parallel_do<Cell, &Cell::prepare>();
	return 0;
}

extern "C" int Fish_Fish_prepare(){
	allocator_handle->parallel_do<Fish, &Fish::prepare>();
	return 0;
}

extern "C" int Cell_Cell_decide(){
	allocator_handle->parallel_do<Cell, &Cell::decide>();
	return 0;
}

extern "C" int Fish_Fish_update(){
	allocator_handle->parallel_do<Fish, &Fish::update>();
	return 0;
}

extern "C" int Shark_Shark_prepare(){
	allocator_handle->parallel_do<Shark, &Shark::prepare>();
	return 0;
}

extern "C" int Shark_Shark_update(){
	allocator_handle->parallel_do<Shark, &Shark::update>();
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