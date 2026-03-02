
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "kernels.h"
#include <stdio.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include "./cub/cub.cuh"



__global__ void gen_mrr_sets_ic_small(
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
							)
{
	__shared__ uint32_t last_rr_num;
	__shared__ uint32_t cur_rr_num;
	__shared__ uint32_t start_pos, end_pos;
	__shared__ uint32_t slide_win[win_size];
    __shared__ uint32_t slide_head, slide_tail;
	__shared__ uint32_t slide_start_pos;
	__shared__ uint32_t root_num;
	const uint32_t tid = blockIdx.x * blockDim.x +threadIdx.x;
	curandStateXORWOW_t local_state;
	curand_init(seed, tid, 0, &local_state);
	visited_flags = (uint32_t*)((char*)visited_flags + blockIdx.x*visited_flags_pitch);

	if(threadIdx.x == 0){
		last_rr_num = last_req_rr_set / gridDim.x  + ((last_req_rr_set % gridDim.x) > blockIdx.x ? 1 : 0);
		cur_rr_num = n_req_rr_sets / gridDim.x + ((n_req_rr_sets % gridDim.x) > blockIdx.x ? 1 : 0);
	}
	__syncwarp();

	while(last_rr_num < cur_rr_num){

		if(threadIdx.x == 0){
			root_num = left_n/left_eta + (1.0*(left_n % left_eta)/left_eta > curand_uniform(&local_state)?1:0);
			start_pos = rr_block_end[ blockIdx.x ];
			rr_start[ blockIdx.x * block_length + last_rr_num] = start_pos;
			rr_root_num[ blockIdx.x * block_length + last_rr_num] = root_num;
			end_pos = start_pos + root_num;
			slide_head = 0;
			slide_start_pos = start_pos;
			slide_tail = root_num < win_size ? root_num : win_size;
		}
		__syncwarp();
		// clear all visited flags
		for(uint32_t i = threadIdx.x; i < (n_nodes+31)/32; i += blockDim.x){
			visited_flags[i] = 0;
		}

		// activate the nodes in the active set
		for(uint32_t i = threadIdx.x; i < n_nodes; i += blockDim.x){
			if (active_set[i]){
				atomicOr(&visited_flags[i/32], 1 << (i%32));
			}
		}
		__syncwarp();
		
		for(uint32_t i = threadIdx.x; i < root_num; i += blockDim.x){
			uint32_t cur_node = ((uint64_t)curand(&local_state) * n_nodes) >> 32;
			while ((atomicOr(&visited_flags[cur_node / 32], 1u << (cur_node % 32)) & (1u << (cur_node % 32))) != 0)
			{
				cur_node = ((uint64_t)curand(&local_state) * n_nodes) >> 32;
			}
			rr_sets[ blockIdx.x * rr_length + (start_pos + i) % rr_length] = cur_node;
			if (i < slide_tail)
				slide_win[i] = cur_node;
			atomicAdd(&visited_cnt[cur_node], 1);
		}
		__syncwarp();
		
		while(slide_head < slide_tail){
			uint32_t froniter = slide_win[slide_head];
			uint32_t offset = offsets[froniter];
			uint32_t n_adjacent = offsets[froniter + 1] - offset;
			uint32_t n_iteration = (n_adjacent + blockDim.x - 1) / blockDim.x * blockDim.x;

			for(uint32_t i = threadIdx.x; i < n_iteration; i += blockDim.x){
				bool active_iter = i < n_adjacent;
				float edge_prob;
				if(active_iter){
					edge_prob = probs[offset + i];
				}
				else{
					edge_prob = 0.0f; 
				}
				if( (active_iter == true) && (curand_uniform(&local_state) < edge_prob) ){
					uint32_t dst = dests[offset + i];
					if( (visited_flags[dst/32] & (1 << (dst%32))) == 0 ){
						atomicOr(&visited_flags[dst/32], 1 << (dst%32));
						uint32_t pos = atomicAdd(&end_pos, 1); // warp premivites to avoid atomic 
						if(abs(int(pos - rr_start[ blockIdx.x * block_length])) % rr_length == 0){
							overflow[0] = 1;
						}
						else{
							rr_sets[ blockIdx.x * rr_length + pos % rr_length] = dst;
							atomicAdd(&visited_cnt[dst], 1);
							uint32_t slide_pos = atomicAdd(&slide_tail, 1);
							if(slide_pos < win_size){
								slide_win[slide_pos] = dst;
							}
						}
					}
				}
				__syncwarp();
			}
			
			if(overflow[0]==1){
				break;
			}

			if(threadIdx.x == 0){
				slide_head++;
				if (slide_head == slide_tail || slide_head == win_size){
					slide_head = 0;
					slide_start_pos = slide_start_pos + (slide_tail > win_size ? win_size : slide_tail);
					slide_tail = (end_pos - slide_start_pos) > win_size ? win_size : (end_pos - slide_start_pos);
				}
				if(slide_start_pos == end_pos){
					slide_head = slide_tail; // break the while loop
				}
			}
			__syncwarp();
			if(slide_head == 0){
				for(uint32_t i = threadIdx.x; i < slide_tail; i+= blockDim.x){
					slide_win[i] = rr_sets[ blockIdx.x * rr_length + (slide_start_pos + i) % rr_length];
				}
			}
			__syncwarp();
		}
		
		if(threadIdx.x == 0){
			rr_block_end[ blockIdx.x ] = blockIdx.x * rr_length + end_pos % rr_length;
			last_rr_num += 1;
			atomicAdd(n_rr_sets, 1);
		}
		__syncwarp();
	}
}

__global__ void gen_mrr_sets_ic_large(
								uint32_t* offsets, uint32_t* dests, const float* probs, bool* active_set,
								uint32_t* visited_flags, const size_t visited_flags_pitch,
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
							)
{	
    __shared__ uint32_t last_rr_num;
    __shared__ uint32_t cur_rr_num;
    __shared__ uint32_t start_pos, end_pos;
    __shared__ uint32_t slide_win[win_size]; 
    __shared__ uint32_t slide_head, slide_tail;
    __shared__ uint32_t slide_start_pos;
	__shared__ uint32_t root_num;
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t lane_id = threadIdx.x % 32;
	curandStateXORWOW_t local_state;
	curand_init(seed, tid, 0, &local_state);
    visited_flags = (uint32_t*)((char*)visited_flags + blockIdx.x * visited_flags_pitch);

    if (threadIdx.x == 0) {
        last_rr_num = last_req_rr_set / gridDim.x + ((last_req_rr_set % gridDim.x) > blockIdx.x ? 1 : 0);
        cur_rr_num = n_req_rr_sets / gridDim.x + ((n_req_rr_sets % gridDim.x) > blockIdx.x ? 1 : 0);
    }
    __syncthreads();

    while (last_rr_num < cur_rr_num) {

        if (threadIdx.x == 0) {
			root_num = left_n/left_eta + (1.0*(left_n % left_eta)/left_eta > curand_uniform(&local_state)?1:0);
            start_pos = rr_block_end[blockIdx.x];
            rr_start[blockIdx.x * block_length + last_rr_num] = start_pos; 
			rr_root_num[ blockIdx.x * block_length + last_rr_num] = root_num;
            end_pos = start_pos + root_num;
            slide_head = 0;
            slide_start_pos = start_pos;
            slide_tail = root_num < win_size ? root_num : win_size;
        }
        __syncthreads();
        // clear all visited flags
		for(uint32_t i = threadIdx.x; i < (n_nodes+31)/32; i += blockDim.x){
			visited_flags[i] = 0;
		}

        // activate the nodes in the active set
        for (uint32_t i = threadIdx.x; i < n_nodes; i += blockDim.x) {
            if (active_set[i]) {
                atomicOr(&visited_flags[i / 32], 1 << (i % 32));
            }
        }
        __syncthreads();

		for(uint32_t i = threadIdx.x; i < root_num; i += blockDim.x){
			uint32_t cur_node = ((uint64_t)curand(&local_state) * n_nodes) >> 32;
			while ((atomicOr(&visited_flags[cur_node / 32], 1u << (cur_node % 32)) & (1u << (cur_node % 32))) != 0)
			{
				cur_node = ((uint64_t)curand(&local_state) * n_nodes) >> 32;
			}
			rr_sets[ blockIdx.x * rr_length + (start_pos + i) % rr_length] = cur_node;
			if (i < slide_tail)
				slide_win[i] = cur_node;
			atomicAdd(&visited_cnt[cur_node], 1);
		}
		__syncthreads();

        while (slide_head < slide_tail) {
			while(true){
				uint32_t my_idx = 0;
				uint32_t has_work = 0;
				if (lane_id == 0) { 
					my_idx = atomicAdd(&slide_head, 1);
					has_work = (my_idx < slide_tail);
				}
				__syncwarp(); 
				my_idx = __shfl_sync(0xffffffff, my_idx, 0);
				has_work = __shfl_sync(0xffffffff, has_work, 0);

				if (!has_work) {
					break;  
				}
				uint32_t my_frontier = slide_win[my_idx];
				uint32_t offset = offsets[my_frontier];
				uint32_t n_adjacent = offsets[my_frontier + 1] - offset;
				uint32_t n_iteration = (n_adjacent + 32 - 1) / 32;  
				for (uint32_t iter = 0; iter < n_iteration; ++iter) {
					uint32_t i = iter * 32 + lane_id;
					bool active_iter = i < n_adjacent;
					float edge_prob = active_iter ? probs[offset + i] : 0.0f;
					if (active_iter && (curand_uniform(&local_state) < edge_prob))
					{
						uint32_t dst = dests[offset + i]; 
						if ((atomicOr(&visited_flags[dst/32], 1u<<(dst%32)) & (1u<<(dst%32))) == 0){
							uint32_t pos = atomicAdd(&end_pos, 1);
							if (abs(int(pos - rr_start[blockIdx.x * block_length]))% rr_length == 0)
							{
								overflow[0] = 1;
							} 
							else
							{
								rr_sets[blockIdx.x * rr_length + pos % rr_length] = dst;
								atomicAdd(&visited_cnt[dst], 1);
							}
						}
					}
					__syncwarp();
				}
			}
            __syncthreads(); 

			if(overflow[0]==1){
				break;
			}

            if (threadIdx.x == 0) {
				slide_head = 0;
				slide_start_pos += slide_tail;  
				slide_tail = (end_pos - slide_start_pos) > win_size ? win_size : (end_pos - slide_start_pos);
            }
            __syncthreads();

            if (slide_head == 0 && slide_tail > 0) {
                for (uint32_t i = threadIdx.x; i < slide_tail; i += blockDim.x) {
                    slide_win[i] = rr_sets[blockIdx.x * rr_length + (slide_start_pos + i) % rr_length];
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            rr_block_end[blockIdx.x] = blockIdx.x * rr_length + end_pos % rr_length;
            last_rr_num += 1;
            atomicAdd(n_rr_sets, 1);
        }
        __syncthreads();
    }
}


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
							)
{

	__shared__ bool found_node;
	__shared__ uint32_t cur_rr_num;
	__shared__ uint32_t rr_id;
	__shared__ uint32_t start_pos, end_pos;
	__shared__ uint32_t len;

	if(threadIdx.x == 0){
		rr_id = 0;
		cur_rr_num = n_req_rr_sets / gridDim.x + ((n_req_rr_sets % gridDim.x) > blockIdx.x ? 1 : 0);
	}
	__syncthreads();

	while(rr_id < cur_rr_num){ 

		if( threadIdx.x == 0){
			start_pos = rr_start[ blockIdx.x * block_length + rr_id ];
			if( rr_id + 1 == cur_rr_num ){
				end_pos = rr_block_end[ blockIdx.x ];
			}
			else{
				end_pos = rr_start[ blockIdx.x * block_length + rr_id + 1];
			}
			found_node = false;
			len = end_pos > start_pos ? end_pos - start_pos : (rr_length - (start_pos - end_pos));
		}
		__syncthreads();

		if( !covered_flags[ blockIdx.x * block_length + rr_id ] )
		{
			for(int i = threadIdx.x; i < len; i += blockDim.x){
				uint32_t node = rr_sets[ blockIdx.x * rr_length + (start_pos + i) % rr_length];
				if( node == last_node_added ){
					found_node = true;
				}
				if(found_node){
					break;
				}
			}
			__syncthreads();
			if(found_node){
				for(int i = threadIdx.x; i < len; i += blockDim.x){
					uint32_t node = rr_sets[ blockIdx.x * rr_length + (start_pos + i) % rr_length];
					atomicAdd(&visited_cnt[node], -1);
				}	
				__syncthreads();
			}
		}
		
		if( threadIdx.x == 0){
			if(found_node){
				covered_flags[ blockIdx.x * block_length + rr_id ] = true;
				covered_flags_num[ blockIdx.x ] += 1;
			}
			rr_id += 1;
		}
		__syncthreads();
	}
}


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
								int beta,
								uint32_t* overflow
							)
{
	__shared__ int update_rr_id, n_rr_block, rr_id;
	__shared__ uint32_t pre_start, pre_end, len;
	__shared__ uint32_t start_pos, end_pos;
	__shared__ uint32_t slide_win[win_size];
	__shared__ uint32_t blockID;
	__shared__ int new_root_num, idle_root_num, pre_root_num;
	curandStateXORWOW_t local_state;
    __shared__ uint32_t slide_head, slide_tail;
	__shared__ uint32_t slide_start_pos;
	const uint32_t tid = blockIdx.x * blockDim.x +threadIdx.x;
	const uint32_t lane_id = threadIdx.x % 32;
	curand_init(seed, tid, 0, &local_state);
	visited_flags = (uint32_t*)((char*)visited_flags + blockIdx.x*visited_flags_pitch);
	int par = 0;

	while(par < beta){

		if(threadIdx.x == 0){
			blockID = ID_block[ blockIdx.x + N_block*par/beta ];
			n_rr_block = n_req_rr_sets / N_block + ((n_req_rr_sets % N_block) > blockID ? 1 : 0);
			update_rr_id = n_rr_block - 1;
			rr_id = 0;
			start_pos = rr_start[ blockID * block_length + update_rr_id];
			pre_start = start_pos;
			pre_end = rr_block_end[ blockID ];
		}
		__syncthreads();

		while(rr_id < n_rr_block)
		{
			// clear all visited flags
			for(uint32_t i = threadIdx.x; i < (n_nodes+31)/32; i += blockDim.x){
				visited_flags[i] = 0;
			}
			// activate the nodes in the active set
			for(uint32_t i = threadIdx.x; i < n_nodes; i += blockDim.x){
				if (active_set[i]){
					atomicOr(&visited_flags[i / 32], 1 << (i % 32));
				}
			}
			__syncthreads();
			
			if( !covered_flags[ blockID * block_length + update_rr_id ]) //clean
			{
				if(threadIdx.x == 0){
					len = pre_end > pre_start ? pre_end - pre_start: (rr_length - (pre_start - pre_end));
					slide_start_pos = blockID * rr_length + (start_pos + len)% rr_length;
					slide_head = 0;
					pre_root_num = rr_root_num[ blockID * block_length + update_rr_id];
					new_root_num = left_n/left_eta + (1.0*(left_n % left_eta)/left_eta > curand_uniform(&local_state)?1:0);
					rr_root_num[ blockID * block_length + update_rr_id] = new_root_num;
					idle_root_num = (len-pre_root_num)*(new_root_num-pre_root_num)/(left_n-pre_root_num) + (float((((len-pre_root_num)*(new_root_num-pre_root_num))%(left_n-pre_root_num))/(left_n-pre_root_num))) > curand_uniform(&local_state)?1:0;
					new_root_num -= idle_root_num + pre_root_num;
					if(new_root_num <= 0) new_root_num = 0;
					slide_tail = new_root_num < win_size ? new_root_num : win_size;
					end_pos = slide_start_pos + new_root_num;
					atomicAdd(n_add_root,1);
				}
				__syncthreads();

				//copy nodes first
				for(uint32_t i = threadIdx.x; i < len; i += blockDim.x){
					uint32_t cur_node = rr_sets[ blockID * rr_length + (pre_start + i) % rr_length ];
					if(new_root_num > 0){
						atomicOr(&visited_flags[cur_node / 32], 1 << (cur_node % 32));
					}
					rr_sets[ blockID * rr_length + (start_pos + i) % rr_length ] = cur_node;
				}
				__syncthreads();
				
				//random swap idle roots
				for(uint32_t i = threadIdx.x; i < idle_root_num; i += blockDim.x){
					uint32_t j = ((uint64_t)curand(&local_state) * (len-pre_root_num)) >> 32;
					if (i != j){
						uint32_t old = atomicExch(&rr_sets[blockID * rr_length + (start_pos + pre_root_num + j) % rr_length], rr_sets[blockID * rr_length + (start_pos + pre_root_num + i) % rr_length]);
						atomicExch(&rr_sets[blockID * rr_length + (start_pos + pre_root_num + i) % rr_length], old);
					}

				}
				__syncthreads();

				//generate new roots 
				for(uint32_t i = threadIdx.x; i < new_root_num; i += blockDim.x){
					uint32_t cur_node = ((uint64_t)curand(&local_state) * n_nodes) >> 32;
					while ((atomicOr(&visited_flags[cur_node / 32], 1u << (cur_node % 32)) & (1u << (cur_node % 32))) != 0)
					{
						cur_node = ((uint64_t)curand(&local_state) * n_nodes) >> 32;
					}
					rr_sets[ blockID * rr_length + (slide_start_pos + i) % rr_length] = cur_node;
					if (i < slide_tail)
						slide_win[i] = cur_node;
					atomicAdd(&visited_cnt[cur_node], 1);
				}
				__syncthreads();
			}

			else //polluted
			{
				if(threadIdx.x == 0){
					new_root_num = left_n/left_eta + (1.0*(left_n % left_eta)/left_eta > curand_uniform(&local_state)?1:0);
					pre_root_num = rr_root_num[ blockID * block_length + update_rr_id];
					rr_root_num[ blockID * block_length + update_rr_id] = new_root_num;
					end_pos = start_pos + new_root_num;
					slide_head = 0;
					slide_start_pos = start_pos;
					slide_tail = new_root_num < win_size ? new_root_num : win_size;
					atomicAdd(n_regen_rr,1);
				}
				__syncthreads();

				//generate new roots and retain old ones
				for(uint32_t i = threadIdx.x; i < new_root_num; i += blockDim.x){
					uint32_t cur_node;
					bool activated = false;
					if(i < pre_root_num){
						cur_node = rr_sets[ blockID * rr_length + (pre_start + i) % rr_length ];
						atomicOr(&visited_flags[cur_node / 32], 1 << (cur_node % 32));
						if(active_set[cur_node]) activated = true;
					}
					if(activated || i >= pre_root_num){
						cur_node = ((uint64_t)curand(&local_state) * n_nodes) >> 32;
						while ((atomicOr(&visited_flags[cur_node / 32], 1u << (cur_node % 32)) & (1u << (cur_node % 32))) != 0)
						{
							cur_node = ((uint64_t)curand(&local_state) * n_nodes) >> 32;
						}
					}
					rr_sets[ blockID * rr_length + (slide_start_pos + i) % rr_length] = cur_node;
					if (i < slide_tail)
						slide_win[i] = cur_node;
					atomicAdd(&visited_cnt[cur_node], 1);
				}
				__syncthreads();
			}	
		
			while (slide_head < slide_tail) {
				while(true){
					uint32_t my_idx = 0;
					bool has_work = false;
					if (lane_id == 0) { 
						my_idx = atomicAdd(&slide_head, 1);
						has_work = (my_idx < slide_tail);
					}
					__syncwarp(); 
					my_idx = __shfl_sync(0xffffffff, my_idx, 0);
					has_work = __shfl_sync(0xffffffff, has_work, 0);

					if (!has_work) {
						break;  
					}
					uint32_t my_frontier = slide_win[my_idx];
					uint32_t offset = offsets[my_frontier];
					uint32_t n_adjacent = offsets[my_frontier + 1] - offset;
					uint32_t n_iteration = (n_adjacent + 32 - 1) / 32;  
					for (uint32_t iter = 0; iter < n_iteration; ++iter) {
						uint32_t i = iter * 32 + lane_id;
						bool active_iter = i < n_adjacent;
						float edge_prob = active_iter ? probs[offset + i] : 0.0f;
						if (active_iter && (curand_uniform(&local_state) < edge_prob))
						{
							uint32_t dst = dests[offset + i];
							if ((atomicOr(&visited_flags[dst/32], 1u<<(dst%32)) & (1u<<(dst%32))) == 0){
								uint32_t pos = atomicAdd(&end_pos, 1);
								if(abs(int(pos - rr_start[blockID * block_length + (update_rr_id + 1) % n_rr_block]))%rr_length == 0)
								{
									overflow[0] = 1;
								} 
								else 
								{
									rr_sets[blockID * rr_length + pos % rr_length] = dst;
									atomicAdd(&visited_cnt[dst], 1);
								}
							}
						}
						__syncwarp(); 
					}
				}
				__syncthreads(); 

				if(overflow[0]==1){
					break;
				}

				if (threadIdx.x == 0) {
					if (slide_head >= slide_tail) {
						slide_head = 0;
						slide_start_pos = slide_start_pos + (slide_tail > win_size ? win_size : slide_tail); 
						slide_tail = (end_pos - slide_start_pos) > win_size ? win_size : (end_pos - slide_start_pos);
						if (slide_start_pos >= end_pos) {
							slide_tail = 0; 
						}	
					}
				}
				__syncthreads();

				if (slide_head == 0 && slide_tail > 0) {
					for (uint32_t i = threadIdx.x; i < slide_tail; i += blockDim.x) {
						slide_win[i] = rr_sets[blockID * rr_length + (slide_start_pos + i) % rr_length];
					}
				}
				__syncthreads();
			}

			if( !covered_flags[ blockID * block_length + update_rr_id ])
			{
				//swap new roots 
				for(uint32_t i = threadIdx.x; i < new_root_num; i += blockDim.x){
					uint32_t old = atomicExch(&rr_sets[blockID * rr_length + (start_pos + idle_root_num + i) % rr_length], rr_sets[blockID * rr_length + (start_pos + len + i) % rr_length]);
					atomicExch(&rr_sets[blockID * rr_length + (start_pos + len + i) % rr_length], old);
				}
				__syncthreads();
			}

			if(threadIdx.x == 0){
				update_rr_id = (update_rr_id + 1) % n_rr_block;
				pre_start = rr_start[ blockID * block_length + update_rr_id];
				pre_end = rr_start[ blockID  * block_length + update_rr_id + 1];
				rr_start[ blockID  * block_length + rr_id] = start_pos;
				rr_id += 1;
				start_pos = blockID  * rr_length + end_pos % rr_length;
				rr_block_end[ blockID  ] = start_pos;
			}
			__syncthreads();
		}
		par+=1;
	}
}


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
							)
{
	__shared__ bool found_node;
	__shared__ uint32_t cur_rr_num;
	__shared__ uint32_t rr_id;
	__shared__ uint32_t start_pos, end_pos;
	__shared__ uint32_t len;

	if(threadIdx.x == 0){
		rr_id = 0;
		cur_rr_num = n_req_rr_sets / gridDim.x + ((n_req_rr_sets % gridDim.x) > blockIdx.x ? 1 : 0);
	}
	__syncthreads();

	while(rr_id < cur_rr_num){ 
		
		if( threadIdx.x == 0){
			start_pos = rr_start[ blockIdx.x * block_length + rr_id ];
			if( rr_id + 1 == cur_rr_num ){
				end_pos = rr_block_end[ blockIdx.x ];
			}
			else{
				end_pos = rr_start[ blockIdx.x * block_length + rr_id + 1];
			}
			found_node = false;
			len = end_pos > start_pos ? end_pos - start_pos : (rr_length - (start_pos - end_pos));
		}
		__syncthreads();

		if( !covered_flags[ blockIdx.x * block_length + rr_id ] )
		{
			for(int i = threadIdx.x; i < len; i += blockDim.x){
				uint32_t node = rr_sets[ blockIdx.x * rr_length + (start_pos + i) % rr_length];
				if( active_set[node] == true ){
					found_node = true;
				}
				if(found_node){
					break;
				}
			}
			__syncthreads();
			if(found_node){
				for(int i = threadIdx.x; i < len; i += blockDim.x){
					uint32_t node = rr_sets[ blockIdx.x * rr_length + (start_pos + i) % rr_length];
					atomicAdd(&visited_cnt[node], -1);
				}
				__syncthreads();	
			}
		}
		
		if( threadIdx.x == 0){
			if(found_node){
				covered_flags[ blockIdx.x * block_length + rr_id ] = true;
				covered_flags_num[ blockIdx.x ] += 1;
			}
			rr_id += 1;
		}
		__syncthreads();
	}
}


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
							)
{
	__shared__ uint32_t last_rr_num;
	__shared__ uint32_t cur_rr_num;
	__shared__ uint32_t start_pos, end_pos;
	__shared__ uint32_t slide_win[win_size];
    __shared__ uint32_t slide_head, slide_tail;
	__shared__ uint32_t slide_start_pos;
	__shared__ uint32_t root_num;
	const uint32_t tid = blockIdx.x * blockDim.x +threadIdx.x;
	curandStateXORWOW_t local_state;
	curand_init(seed, tid, 0, &local_state);
	visited_flags = (uint32_t*)((char*)visited_flags + blockIdx.x*visited_flags_pitch);

	if(threadIdx.x == 0){
		last_rr_num = last_req_rr_set / gridDim.x  + ((last_req_rr_set % gridDim.x) > blockIdx.x ? 1 : 0);
		cur_rr_num = n_req_rr_sets / gridDim.x + ((n_req_rr_sets % gridDim.x) > blockIdx.x ? 1 : 0);
	}
	__syncthreads();

	while(last_rr_num < cur_rr_num){

		if(threadIdx.x == 0){
			root_num = left_n/left_eta + (1.0*(left_n % left_eta)/left_eta > curand_uniform(&local_state)?1:0);
			start_pos = rr_block_end[ blockIdx.x ];
			rr_start[ blockIdx.x * block_length + last_rr_num] = start_pos;
			rr_root_num[ blockIdx.x * block_length + last_rr_num] = root_num;
			end_pos = start_pos + root_num;
			slide_head = 0;
			slide_start_pos = start_pos;
			slide_tail = root_num < win_size ? root_num : win_size;
		}
		__syncthreads();
		// clear all visited flags
		for(uint32_t i = threadIdx.x; i < (n_nodes+31)/32; i += blockDim.x){
			visited_flags[i] = 0;
		}

		// activate the nodes in the active set
		for(uint32_t i = threadIdx.x; i < n_nodes; i += blockDim.x){
			if (active_set[i]){
				atomicOr(&visited_flags[i/32], 1 << (i%32));
			}
		}
		__syncthreads();
		
		for(uint32_t i = threadIdx.x; i < root_num; i += blockDim.x){
			uint32_t cur_node = ((uint64_t)curand(&local_state) * n_nodes) >> 32;
			while ((visited_flags[cur_node / 32] & (1 << (cur_node % 32))) != 0) {
				cur_node = ((uint64_t)curand(&local_state) * n_nodes) >> 32;
			}
			atomicOr(&visited_flags[cur_node / 32], 1 << (cur_node % 32)); //check there is confict or not
			rr_sets[ blockIdx.x * rr_length + (start_pos + i) % rr_length] = cur_node;
			if (i < slide_tail)
				slide_win[i] = cur_node;
			atomicAdd(&visited_cnt[cur_node], 1);
		}
		__syncthreads();
		
		while(slide_head < slide_tail)
        {
			
			for(uint32_t i = threadIdx.x; i < slide_tail; i += blockDim.x){
                uint32_t froniter = slide_win[i];
                uint32_t offset = offsets[froniter];
                uint32_t n_adjacent = offsets[froniter + 1] - offset;
				if(n_adjacent > 0){
					uint32_t index = ((uint64_t)curand(&local_state) * n_adjacent) >> 32;
					uint32_t dst = dests[offset + index];
					if ((atomicOr(&visited_flags[dst/32], 1u<<(dst%32)) & (1u<<(dst%32))) == 0){
						uint32_t pos = atomicAdd(&end_pos, 1); 
						if(abs(int(pos - rr_start[ blockIdx.x * block_length])) % rr_length == 0){
							overflow[0] = 1;
						}
						else
						{
							rr_sets[blockIdx.x * rr_length + pos % rr_length] = dst;
							atomicAdd(&visited_cnt[dst], 1);
						}
					}
				}
            }
			__syncthreads();
			if(overflow[0]==1){
				break;
			}
			if(threadIdx.x == 0){
                slide_head = 0;
                slide_start_pos = slide_start_pos + (slide_tail > win_size ? win_size : slide_tail);
                slide_tail = (end_pos - slide_start_pos) > win_size ? win_size : (end_pos - slide_start_pos);
				if(slide_start_pos == end_pos){
					slide_head = slide_tail; // break the while loop
				}
			}
			__syncthreads();
			if(slide_head == 0){
				for(uint32_t i = threadIdx.x; i < slide_tail; i+= blockDim.x){
					slide_win[i] = rr_sets[ blockIdx.x * rr_length + (slide_start_pos + i) % rr_length];
				}
			}
			__syncthreads();
		}
		
		if(threadIdx.x == 0){
			rr_block_end[ blockIdx.x ] = blockIdx.x * rr_length + end_pos % rr_length;
			last_rr_num += 1;
			atomicAdd(n_rr_sets, 1);
		}
		__syncthreads();
	}
}


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
								int beta,
								uint32_t* overflow
							)
{
	__shared__ int update_rr_id, n_rr_block, rr_id;
	__shared__ uint32_t pre_start, pre_end, len;
	__shared__ uint32_t start_pos, end_pos;
	__shared__ uint32_t slide_win[win_size];
	__shared__ uint32_t blockID;
	__shared__ int new_root_num, idle_root_num, pre_root_num;
	curandStateXORWOW_t local_state;
    __shared__ uint32_t slide_head, slide_tail;
	__shared__ uint32_t slide_start_pos;
	const uint32_t tid = blockIdx.x * blockDim.x +threadIdx.x;
	// const uint32_t lane_id = threadIdx.x % 32;
	curand_init(seed, tid, 0, &local_state);
	visited_flags = (uint32_t*)((char*)visited_flags + blockIdx.x*visited_flags_pitch);
	int par = 0;

	while(par < beta){

		if(threadIdx.x == 0){
			blockID = ID_block[ blockIdx.x + N_block*par/beta ];
			n_rr_block = n_req_rr_sets / N_block + ((n_req_rr_sets % N_block) > blockID ? 1 : 0);
			update_rr_id = n_rr_block - 1;
			rr_id = 0;
			start_pos = rr_start[ blockID * block_length + update_rr_id];
			pre_start = start_pos;
			pre_end = rr_block_end[ blockID ];
		}
		__syncthreads();

		while(rr_id < n_rr_block)
		{
			// clear all visited flags
			for(uint32_t i = threadIdx.x; i < (n_nodes+31)/32; i += blockDim.x){
				visited_flags[i] = 0;
			}
			// activate the nodes in the active set
			for(uint32_t i = threadIdx.x; i < n_nodes; i += blockDim.x){
				if (active_set[i]){
					atomicOr(&visited_flags[i / 32], 1 << (i % 32));
				}
			}
			__syncthreads();
			
			if( !covered_flags[ blockID * block_length + update_rr_id ]) //clean
			{
				if(threadIdx.x == 0){
					len = pre_end > pre_start ? pre_end - pre_start: (rr_length - (pre_start - pre_end));
					slide_start_pos = blockID * rr_length + (start_pos + len)% rr_length;
					slide_head = 0;
					uint32_t pre_root_num = rr_root_num[ blockID * block_length + update_rr_id];
					new_root_num = left_n/left_eta + (1.0*(left_n % left_eta)/left_eta > curand_uniform(&local_state)?1:0);
					rr_root_num[ blockID * block_length + update_rr_id] = new_root_num;
					idle_root_num = (len-pre_root_num)*(new_root_num-pre_root_num)/(left_n-pre_root_num) + (float((((len-pre_root_num)*(new_root_num-pre_root_num))%(left_n-pre_root_num))/(left_n-pre_root_num))) > curand_uniform(&local_state)?1:0;
					new_root_num -= idle_root_num + pre_root_num;
					if(new_root_num <= 0) new_root_num = 0;
					slide_tail = new_root_num < win_size ? new_root_num : win_size;
					end_pos = slide_start_pos + new_root_num;
					atomicAdd(n_add_root,1);
				}
				__syncthreads();

				//copy nodes first
				for(uint32_t i = threadIdx.x; i < len; i += blockDim.x){
					uint32_t cur_node = rr_sets[ blockID * rr_length + (pre_start + i) % rr_length ];
					if(new_root_num > 0){
						atomicOr(&visited_flags[cur_node / 32], 1 << (cur_node % 32));
					}
					rr_sets[ blockID * rr_length + (start_pos + i) % rr_length ] = cur_node;
				}
				__syncthreads();

				//random swap promoted roots
				for(uint32_t i = threadIdx.x; i < idle_root_num; i += blockDim.x){
					uint32_t j = ((uint64_t)curand(&local_state) * (len-pre_root_num)) >> 32;
					if (i != j){
						uint32_t old = atomicExch(&rr_sets[blockID * rr_length + (start_pos + pre_root_num + j) % rr_length], rr_sets[blockID * rr_length + (start_pos + pre_root_num + i) % rr_length]);
						atomicExch(&rr_sets[blockID * rr_length + (start_pos + pre_root_num + i) % rr_length], old);
					}

				}
				__syncthreads();

				//generate new roots 
				for(uint32_t i = threadIdx.x; i < new_root_num; i += blockDim.x){
					uint32_t cur_node = ((uint64_t)curand(&local_state) * n_nodes) >> 32;
					while ((visited_flags[cur_node / 32] & (1 << (cur_node % 32))) != 0) {
						cur_node = ((uint64_t)curand(&local_state) * n_nodes) >> 32;
					}
					atomicOr(&visited_flags[cur_node / 32], 1 << (cur_node % 32)); 
					rr_sets[ blockID * rr_length + (slide_start_pos + i) % rr_length] = cur_node;
					if (i < slide_tail)
						slide_win[i] = cur_node;
					atomicAdd(&visited_cnt[cur_node], 1);
				}
				__syncthreads();
				
			}
			
			else //polluted
			{
				if(threadIdx.x == 0){
					new_root_num = left_n/left_eta + (1.0*(left_n % left_eta)/left_eta > curand_uniform(&local_state)?1:0);
					pre_root_num = rr_root_num[ blockID * block_length + update_rr_id];
					rr_root_num[ blockID * block_length + update_rr_id] = new_root_num;
					end_pos = start_pos + new_root_num;
					slide_head = 0;
					slide_start_pos = start_pos;
					slide_tail = new_root_num < win_size ? new_root_num : win_size;
					atomicAdd(n_regen_rr,1);
				}
				__syncthreads();

				//
				//generate new roots and retain old ones
				for(uint32_t i = threadIdx.x; i < new_root_num; i += blockDim.x){
					uint32_t cur_node;
					bool activated = false;
					if(i < pre_root_num){
						cur_node = rr_sets[ blockID * rr_length + (pre_start + i) % rr_length ];
						atomicOr(&visited_flags[cur_node / 32], 1 << (cur_node % 32));
						if(active_set[cur_node]) activated = true;
					}
					if(activated || i >= pre_root_num){
						cur_node = ((uint64_t)curand(&local_state) * n_nodes) >> 32;
						while ((atomicOr(&visited_flags[cur_node / 32], 1u << (cur_node % 32)) & (1u << (cur_node % 32))) != 0)
						{
							cur_node = ((uint64_t)curand(&local_state) * n_nodes) >> 32;
						}
					}
					rr_sets[ blockID * rr_length + (slide_start_pos + i) % rr_length] = cur_node;
					if (i < slide_tail)
						slide_win[i] = cur_node;
					atomicAdd(&visited_cnt[cur_node], 1);
				}
				__syncthreads();
			}	
			
			while(slide_head < slide_tail)
			{

				for(uint32_t i = threadIdx.x; i < slide_tail; i += blockDim.x){
					uint32_t froniter = slide_win[i];
					uint32_t offset = offsets[froniter];
					uint32_t n_adjacent = offsets[froniter + 1] - offset;
					if(n_adjacent > 0){
						uint32_t index = ((uint64_t)curand(&local_state) * n_adjacent) >> 32;
						uint32_t dst = dests[offset + index];
						if ((atomicOr(&visited_flags[dst/32], 1u<<(dst%32)) & (1u<<(dst%32))) == 0){
							uint32_t pos = atomicAdd(&end_pos, 1); 
							if(abs(int(pos - rr_start[blockID * block_length + (update_rr_id + 1)% n_rr_block]))%rr_length == 0){
								overflow[0] = 1;
							}
							else
							{
								rr_sets[blockID * rr_length + pos % rr_length] = dst;
								atomicAdd(&visited_cnt[dst], 1);
							}
						}
					}
				}
				__syncthreads();

				if(overflow[0]==1){
					break;
				}
				if(threadIdx.x == 0){
					slide_head = 0;
					slide_start_pos = slide_start_pos + (slide_tail > win_size ? win_size : slide_tail);
					slide_tail = (end_pos - slide_start_pos) > win_size ? win_size : (end_pos - slide_start_pos);
					if(slide_start_pos >= end_pos){
						slide_tail=0; // break the while loop
					}
				}
				__syncthreads();
				if(slide_head == 0 && slide_tail > 0){
					for(uint32_t i = threadIdx.x; i < slide_tail; i+= blockDim.x){
						slide_win[i] = rr_sets[ blockID * rr_length + (slide_start_pos + i) % rr_length];
					}
				}
				__syncthreads();
			}

			if( !covered_flags[ blockID * block_length + update_rr_id ])
			{
				//swap new roots 
				for(uint32_t i = threadIdx.x; i < new_root_num; i += blockDim.x){
					uint32_t old = atomicExch(&rr_sets[blockID * rr_length + (start_pos + idle_root_num + i) % rr_length], rr_sets[blockID * rr_length + (start_pos + len + i) % rr_length]);
					atomicExch(&rr_sets[blockID * rr_length + (start_pos + len + i) % rr_length], old);
				}
				__syncthreads();
			}
				
			if(threadIdx.x == 0){
				update_rr_id = (update_rr_id + 1) % n_rr_block;
				pre_start = rr_start[ blockID * block_length + update_rr_id];
				pre_end = rr_start[ blockID  * block_length + update_rr_id + 1];
				rr_start[ blockID  * block_length + rr_id] = start_pos;
				rr_id += 1;
				start_pos = blockID  * rr_length + end_pos % rr_length;
				rr_block_end[ blockID  ] = start_pos;
			}
			__syncthreads();
		}
		par+=1;
	}
}
