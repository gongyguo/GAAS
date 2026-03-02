
#pragma once

#include <iostream>

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
printf("Error at %s:%d\n",__FILE__,__LINE__);\
return EXIT_FAILURE;}} while(0)


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}


const uint32_t win_size = 512;
const int block_length = 1024;
const int SHARED_MEM_SIZE = 1024;

struct MultiplyStep {
    uint32_t step;
    __host__ __device__ MultiplyStep(uint32_t s) : step(s) {}
    __host__ __device__ uint32_t operator()(uint32_t x) const {
        return x * step;
    }
};

__global__ void gen_mrr_sets_ic_small(
								uint32_t* offsets, uint32_t* dests, const float* probs, bool* active_set,
								uint32_t* visited_flags, const size_t visited_flags_pitch,
								uint32_t seed, const int n_nodes,
								const uint32_t last_req_rr_set,
								const uint32_t n_req_rr_sets,
								const uint32_t rr_length,
								uint32_t* root_num,
								const uint32_t left_eta,
								const uint32_t left_n,
								uint32_t* rr_sets,
								float* visited_cnt,
								uint32_t* n_rr_sets,
								uint32_t* rr_start,
								uint32_t* rr_block_end,
								uint32_t* overflow
							);

__global__ void gen_mrr_sets_ic_large(
								uint32_t* offsets, uint32_t* dests, const float* probs, bool* active_set,
								uint32_t* visited_flags, const size_t visited_flags_pitch,
								uint32_t seed, const int n_nodes,
								const uint32_t last_req_rr_set,
								const uint32_t n_req_rr_sets,
								const uint32_t rr_length,
								uint32_t* root_num,
								const uint32_t left_eta,
								const uint32_t left_n,
								uint32_t* rr_sets,
								float* visited_cnt,
								uint32_t* n_rr_sets,
								uint32_t* rr_start,
								uint32_t* rr_block_end,
								uint32_t* overflow
							);

__global__ void cover_mrr_sets(
								int last_node_added, 
								bool* covered_flags,
								int* covered_flags_num,
								float* visited_cnt,
								const uint32_t n_req_rr_sets,
								const uint32_t rr_length,
								uint32_t* rr_sets,
								uint32_t* rr_start,
								uint32_t* rr_block_end
							);
							

__global__ void update_mrr_sets_ic(
								uint32_t* offsets, uint32_t* dests, const float* probs, bool* active_set,
								uint32_t* visited_flags,const size_t visited_flags_pitch,
								uint32_t seed, const int n_nodes,
								const uint32_t n_req_rr_sets,
								const uint32_t rr_length,
								uint32_t* rr_root_num,
								const uint32_t left_eta,
								const uint32_t left_n,
								uint32_t* n_add_root,
            					uint32_t* n_regen_rr,
								uint32_t* rr_sets,
								float* visited_cnt,
								uint32_t* n_rr_sets,
								uint32_t* rr_start,
								uint32_t* rr_block_end,
								bool* covered_flags,
								int* ID_block,
								uint32_t N_block,
								int par,
								uint32_t* overflow
							);

__global__ void discover_polluted_mrr_sets(								
								bool* active_set, 
								bool* covered_flags,
								int* covered_flags_num,
								float* visited_cnt,
								const uint32_t n_req_rr_sets,
								const uint32_t rr_length,
								uint32_t* rr_sets,
								uint32_t* rr_start,
								uint32_t* rr_block_end
							);


__global__ void gen_mrr_sets_lt(
								uint32_t* offsets, uint32_t* dests, const float* probs, bool* active_set,
								uint32_t* visited_flags,const size_t visited_flags_pitch,
								uint32_t seed, const int n_nodes,
								const uint32_t last_req_rr_set,
								const uint32_t n_req_rr_sets,
								const uint32_t rr_length,
								uint32_t* rr_root_num,
								const uint32_t left_eta,
								const uint32_t left_n,
								uint32_t* rr_sets,
								float* visited_cnt,
								uint32_t* n_rr_sets,
								uint32_t* rr_start,
								uint32_t* rr_block_end,
								uint32_t* overflow
							);


__global__ void update_mrr_sets_lt(
								uint32_t* offsets, uint32_t* dests, const float* probs, bool* active_set,
								uint32_t* visited_flags,const size_t visited_flags_pitch,
								uint32_t seed, const int n_nodes,
								const uint32_t n_req_rr_sets,
								const uint32_t rr_length,
								uint32_t* rr_root_num,
								const uint32_t left_eta,
								const uint32_t left_n,
								uint32_t* n_add_root,
            					uint32_t* n_regen_rr,
								uint32_t* rr_sets,
								float* visited_cnt,
								uint32_t* n_rr_sets,
								uint32_t* rr_start,
								uint32_t* rr_block_end,
								bool* covered_flags,
								int* ID_block,
								uint32_t N_block,
								int par,
								uint32_t* overflow
							);