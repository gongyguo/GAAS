
#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <queue>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include "kernels.h"
#include "./cub/cub.cuh"
#include "argument.h"


class Math{
    public:
        static double log2(int n){
            return log(n) / log(2);
        }
        static double logcnk(int n, int k)
        {
            double ans = 0;
            for(int i = n - k + 1; i <= n; i++){
                ans += log(i);
            }
            for(int i = 1; i <= k; i++){
                ans -= log(i);
            }
            return ans;
        }
};

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

class GAAS
{
    private:
        bool IC;
        bool LT;
        double epsilon;
        int n_nodes;
        uint32_t seed = 0;
        int N_RR;
        int N_BLOCKS;
        int left_n = 0;
        int left_eta = 0;
        int active_node = 0;
        std::vector<std::vector<uint32_t>> PO;
        std::vector<float> costs;
        std::vector<uint32_t> seedSet;
        std::vector<uint32_t> offsets;
        std::vector<uint32_t> dests;
        std::vector<float> probs;
        std::vector<bool> active_set;
        thrust::device_vector<float> d_costs;
        thrust::device_vector<uint32_t> d_dests;
        thrust::device_vector<float> d_probs;
        thrust::device_vector<uint32_t> d_offsets;
        thrust::device_vector<bool> d_active_set;
        thrust::device_vector<uint32_t> d_rr_sets;
        thrust::device_vector<uint32_t> d_n_rr_sets;
        thrust::device_vector<uint32_t> d_n_add_root;
        thrust::device_vector<uint32_t> d_n_regen_rr;
        thrust::device_vector<float> d_result;
        thrust::device_vector<float> d_visited_cnt;
        thrust::device_vector<float> d_visited_cnt_temp;
        thrust::device_vector<bool> d_covered_flags;
        thrust::device_vector<int> d_covered_flags_num;
        thrust::device_vector<int> d_covered_indices;
        thrust::device_vector<uint32_t> d_root_num;
        thrust::device_vector<uint32_t> d_rr_block_end;
        thrust::device_vector<uint32_t> d_rr_start;
        thrust::device_vector<uint32_t> d_overflow;
        size_t RRset_mem;
        uint32_t rr_round = 0;
        uint32_t length;
        uint32_t last_root_num = 0;
        uint32_t root_num = 0;
        uint32_t n_exist_rr = 0;
        double build_time = 0;
        double select_time = 0;
        double revise_time = 0;

        uint32_t *d_visited_flags;
        size_t visited_flags_pitch;
        uint32_t gen_mrr_sets(uint32_t last_n_req, uint32_t n_req, bool large);
        void update_mrr_sets(uint32_t n_req, int beta);
    
        public:
        GAAS(std::string filename, const double epsilon, const bool reversed=true, const bool large=false, const bool IC=true);
        ~GAAS();
        void configure_large_block();
        void clear();
        void seedminmization(Argument &arg);
        void AdaptiveSelect(Argument & arg, const double factor, const double epsilon, const double delta);
        int logcnk(int n, int k);
        void RandBatch(int eta);
        void create_csr(std::string filename, const bool reversed);
        void load_costs(const std::string dataset, const int num_nodes);
        void load_realization(int index, const int dataset_No, bool IC);
        void realization(std::vector<uint32_t> batch_set, int &active_node);
        float batch_seed_selection(uint32_t nrr, int batch_size, std::vector<uint32_t> &seed_set);
};