
#include "GAAS.h"
#include <fstream>
#include <vector>
#include <numeric>
#include <random>

GAAS::~GAAS()
{
    HANDLE_ERROR(cudaFree(d_visited_flags));
}

void GAAS::clear(){
    cudaMemset(thrust::raw_pointer_cast(d_n_rr_sets.data()), 0, d_n_rr_sets.size() * sizeof(uint32_t));
    cudaMemset(thrust::raw_pointer_cast(d_rr_start.data()), 0, d_rr_start.size() * sizeof(uint32_t));
    cudaMemset(thrust::raw_pointer_cast(d_rr_sets.data()), 0, d_rr_sets.size() * sizeof(uint32_t));
    cudaMemset(thrust::raw_pointer_cast(d_visited_cnt.data()), 0, d_visited_cnt.size() * sizeof(float));
    thrust::sequence(thrust::device, d_rr_block_end.begin(), d_rr_block_end.end()); 
    thrust::transform(thrust::device, d_rr_block_end.begin(), d_rr_block_end.end(), d_rr_block_end.begin(),
                    MultiplyStep(length)); 
}

GAAS::GAAS(std::string dataset, const double epsilon, const bool reversed,  const bool large, const bool ICLT) : epsilon{epsilon}
{
    IC = ICLT;
    LT = !ICLT;
    create_csr(dataset, reversed);
    n_nodes = offsets.size() - 1;
    load_costs(dataset, n_nodes);
    std::cout << "# of nodes: " << n_nodes << std::endl;
    std::cout << "# of edges: " << dests.size() << std::endl;

    PO.resize(n_nodes, std::vector<uint32_t>());
    active_set = std::vector<bool>(n_nodes, false);

    // Initialize GPU arrays below
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if(!large){
        N_BLOCKS = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor/prop.warpSize); //82 *（1536/32） = 3936
    }
    else{
        N_BLOCKS = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor/128);
        std::cout << "Use large block to generate RR sets " << N_BLOCKS << std::endl;
    }
    N_RR = N_BLOCKS * block_length; //max number of rr sets
    d_costs = costs;
    d_dests = dests;
    d_probs = probs;
    d_offsets = offsets;
    d_active_set = active_set;
    d_n_rr_sets = thrust::device_vector<uint32_t>(1, 0);
    d_n_add_root = thrust::device_vector<uint32_t>(1, 0);
    d_n_regen_rr = thrust::device_vector<uint32_t>(1, 0);
    d_overflow = thrust::device_vector<uint32_t>(1, 0);
    d_result = thrust::device_vector<float>(n_nodes, 0);
    d_visited_cnt = thrust::device_vector<float>(n_nodes, 0);
    d_visited_cnt_temp = thrust::device_vector<int>(n_nodes, 0);
    d_covered_flags = thrust::device_vector<bool>(N_RR, 0);
    d_covered_flags_num = thrust::device_vector<int>(N_BLOCKS, 0);
    d_covered_indices = thrust::device_vector<int>(N_BLOCKS, 0);
    d_root_num = thrust::device_vector<uint32_t>(N_RR, 0); 
    d_rr_start = thrust::device_vector<uint32_t>(N_RR, 0);
    d_rr_block_end = thrust::device_vector<uint32_t>(N_BLOCKS);

    HANDLE_ERROR(cudaMallocPitch((void**)&d_visited_flags, &visited_flags_pitch, (n_nodes+31)/32  * sizeof(uint32_t), N_BLOCKS));
    size_t l_free = 0;
    size_t l_Total = 0;
    HANDLE_ERROR(cudaMemGetInfo(&l_free, &l_Total));
    if (l_free > static_cast<size_t>(16)*1024*1024*1024){
        RRset_mem = static_cast<size_t>(16)*1024*1024*1024;
    }
    else{
        RRset_mem = 0.95 * l_free;
    }
    std::cout << "RRset allocated " << (RRset_mem/1024/1024/1024)<< " GB" <<std::endl;
    d_rr_sets = thrust::device_vector<uint32_t>((RRset_mem/sizeof(uint32_t)), 0);
    length = static_cast<uint32_t>(RRset_mem / (sizeof(uint32_t))/N_BLOCKS);
    thrust::sequence(thrust::device, d_rr_block_end.begin(), d_rr_block_end.end()); 
    thrust::transform(thrust::device, d_rr_block_end.begin(), d_rr_block_end.end(), d_rr_block_end.begin(),
                    MultiplyStep(length)); 
 
    HANDLE_ERROR(cudaMemGetInfo(&l_free, &l_Total));
    std::cout << "Free memory: " << (l_free/1024/1024/1024)<< " GB" <<std::endl;
}

void GAAS::configure_large_block() {

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int num_threads = 128;
    N_BLOCKS = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor/num_threads);
    std::cout << "Use multi-warps kernel to generate mRR-sets " << N_BLOCKS << std::endl;
    thrust::device_vector<uint32_t> temp2;
    d_rr_block_end.swap(temp2);
    d_rr_block_end = thrust::device_vector<uint32_t>(N_BLOCKS);
    length = static_cast<uint32_t>(RRset_mem / (sizeof(uint32_t))/N_BLOCKS);
    clear();
}

float GAAS::batch_seed_selection(uint32_t nrr, int batch_size, std::vector<uint32_t> &batch_set)
{
    float coverage = 0.0;
    cudaMemset(thrust::raw_pointer_cast(d_covered_flags.data()), false, d_covered_flags.size() * sizeof(bool));
    cudaMemset(thrust::raw_pointer_cast(d_covered_flags_num.data()), 0, d_covered_flags_num.size() * sizeof(uint32_t));
    d_visited_cnt_temp = d_visited_cnt;

    for(int k = 0; k < batch_size; k++){
        auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(d_visited_cnt_temp.begin(), d_costs.begin()));
        auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(d_visited_cnt_temp.end(), d_costs.end()));
        thrust::transform(zip_begin, zip_end, d_result.begin(), 
                            [] __host__ __device__ (const thrust::tuple<float, float>& p) {
                                float inf = thrust::get<0>(p);
                                float cost = thrust::get<1>(p);
                                return inf / cost;
                            });
        auto max_iter = thrust::max_element(d_result.begin(), d_result.end());
        int max_index = thrust::distance(d_result.begin(), max_iter);
        float max_value = d_visited_cnt_temp[max_index];
        if (max_value == 0){
            return coverage;
        }
        coverage += max_value;
        batch_set.push_back(max_index);
        cover_mrr_sets<<<N_BLOCKS, 32>>>(
            max_index,
            thrust::raw_pointer_cast(d_covered_flags.data()),
            thrust::raw_pointer_cast(d_covered_flags_num.data()),
            thrust::raw_pointer_cast(d_visited_cnt_temp.data()),
            nrr,
            length,
            thrust::raw_pointer_cast(d_rr_sets.data()),
            thrust::raw_pointer_cast(d_rr_start.data()),
            thrust::raw_pointer_cast(d_rr_block_end.data())
        );
        cudaDeviceSynchronize();

    }
    return coverage;    
}

uint32_t GAAS::gen_mrr_sets(uint32_t last_n_req, uint32_t n_req, bool large)
{
    seed+=1;
    if (IC && (n_req - last_n_req) > 0 && !large){
        gen_mrr_sets_ic_small<<<N_BLOCKS, 32>>>(
            thrust::raw_pointer_cast(d_offsets.data()),
            thrust::raw_pointer_cast(d_dests.data()),
            thrust::raw_pointer_cast(d_probs.data()),
            thrust::raw_pointer_cast(d_active_set.data()),
            d_visited_flags, visited_flags_pitch,
            seed, n_nodes,
            last_n_req,
            n_req,
            length,
            thrust::raw_pointer_cast(d_root_num.data()),
            left_eta,
            left_n,
            thrust::raw_pointer_cast(d_rr_sets.data()),
            thrust::raw_pointer_cast(d_visited_cnt.data()),
            thrust::raw_pointer_cast(d_n_rr_sets.data()),
            thrust::raw_pointer_cast(d_rr_start.data()),
            thrust::raw_pointer_cast(d_rr_block_end.data()),
            thrust::raw_pointer_cast(d_overflow.data())
            );
        cudaDeviceSynchronize();
    }
    else if (IC && large){
        gen_mrr_sets_ic_large<<<N_BLOCKS, 128>>>(
            thrust::raw_pointer_cast(d_offsets.data()),
            thrust::raw_pointer_cast(d_dests.data()),
            thrust::raw_pointer_cast(d_probs.data()),
            thrust::raw_pointer_cast(d_active_set.data()),
            d_visited_flags, visited_flags_pitch,
            seed, n_nodes,
            last_n_req,
            n_req,
            length,
            thrust::raw_pointer_cast(d_root_num.data()),
            left_eta,
            left_n,
            thrust::raw_pointer_cast(d_rr_sets.data()),
            thrust::raw_pointer_cast(d_visited_cnt.data()),
            thrust::raw_pointer_cast(d_n_rr_sets.data()),
            thrust::raw_pointer_cast(d_rr_start.data()),
            thrust::raw_pointer_cast(d_rr_block_end.data()),
            thrust::raw_pointer_cast(d_overflow.data())
            );
        cudaDeviceSynchronize();
    }    
    else if (LT && (n_req - last_n_req) > 0){
        gen_mrr_sets_lt<<<N_BLOCKS, 32>>>(
            thrust::raw_pointer_cast(d_offsets.data()),
            thrust::raw_pointer_cast(d_dests.data()),
            thrust::raw_pointer_cast(d_probs.data()),
            thrust::raw_pointer_cast(d_active_set.data()),
            d_visited_flags, visited_flags_pitch,
            seed, n_nodes,
            last_n_req,
            n_req,
            length,
            thrust::raw_pointer_cast(d_root_num.data()),
            left_eta,
            left_n,
            thrust::raw_pointer_cast(d_rr_sets.data()),
            thrust::raw_pointer_cast(d_visited_cnt.data()),
            thrust::raw_pointer_cast(d_n_rr_sets.data()),
            thrust::raw_pointer_cast(d_rr_start.data()),
            thrust::raw_pointer_cast(d_rr_block_end.data()),
            thrust::raw_pointer_cast(d_overflow.data())
            );
        cudaDeviceSynchronize();
    }
    else{
        std::cout << "Not implemented model!" << std::endl;
    }
    const int n_generated_rr = d_n_rr_sets[0];
    return n_generated_rr;
}

void GAAS::update_mrr_sets(uint32_t n_req, int beta)   
{   
    seed+=1;
    if (IC && n_req > 0){
        update_mrr_sets_ic<<<N_BLOCKS/beta, 32*beta>>>(
            thrust::raw_pointer_cast(d_offsets.data()),
            thrust::raw_pointer_cast(d_dests.data()),
            thrust::raw_pointer_cast(d_probs.data()),
            thrust::raw_pointer_cast(d_active_set.data()),
            d_visited_flags, visited_flags_pitch,
            seed, n_nodes,
            n_req,
            length,
            thrust::raw_pointer_cast(d_root_num.data()),
            left_eta,
            left_n,
            thrust::raw_pointer_cast(d_n_add_root.data()),
            thrust::raw_pointer_cast(d_n_regen_rr.data()),
            thrust::raw_pointer_cast(d_rr_sets.data()),
            thrust::raw_pointer_cast(d_visited_cnt.data()),
            thrust::raw_pointer_cast(d_n_rr_sets.data()),
            thrust::raw_pointer_cast(d_rr_start.data()),
            thrust::raw_pointer_cast(d_rr_block_end.data()),
            thrust::raw_pointer_cast(d_covered_flags.data()),
            thrust::raw_pointer_cast(d_covered_indices.data()),
            N_BLOCKS,
            beta,
            thrust::raw_pointer_cast(d_overflow.data())
            ); 
        cudaDeviceSynchronize();  
    }
    else if (LT && n_req > 0 ){
        update_mrr_sets_lt<<<N_BLOCKS/beta, 32*beta>>>(
            thrust::raw_pointer_cast(d_offsets.data()),
            thrust::raw_pointer_cast(d_dests.data()),
            thrust::raw_pointer_cast(d_probs.data()),
            thrust::raw_pointer_cast(d_active_set.data()),
            d_visited_flags, visited_flags_pitch,
            seed, n_nodes,
            n_req,
            length,
            thrust::raw_pointer_cast(d_root_num.data()),
            left_eta,
            left_n,
            thrust::raw_pointer_cast(d_n_add_root.data()),
            thrust::raw_pointer_cast(d_n_regen_rr.data()),
            thrust::raw_pointer_cast(d_rr_sets.data()),
            thrust::raw_pointer_cast(d_visited_cnt.data()),
            thrust::raw_pointer_cast(d_n_rr_sets.data()),
            thrust::raw_pointer_cast(d_rr_start.data()),
            thrust::raw_pointer_cast(d_rr_block_end.data()),
            thrust::raw_pointer_cast(d_covered_flags.data()),
            thrust::raw_pointer_cast(d_covered_indices.data()),
            N_BLOCKS,
            beta,
            thrust::raw_pointer_cast(d_overflow.data())
            );
        cudaDeviceSynchronize();
    }
    else{
        std::cout << "Not implemented model!" << std::endl;
    }
}

void GAAS::seedminmization(Argument &arg)
{
    int expected_nodes = arg.eta;
    std::cout << "expected spread: " << expected_nodes << std::endl;		                		
    const double factor = 1. - pow(1. - 1. / arg.batch, arg.batch);
    load_realization(arg.start_time, arg.dataset_No, arg.model);			
    active_node = 0;				
    left_eta = expected_nodes; //change the left_node as the quota
    left_n = n_nodes;
    GpuTimer gpu_timer;
    gpu_timer.Start();    
    if(arg.large){
        configure_large_block();
    }
    float r = 0;
    if(IC) r = 0.5;
    if(LT) r = 0.999;
    while (active_node < expected_nodes)  //set the eta as 1;
    {							
        left_eta = expected_nodes - active_node;
        left_n = n_nodes - active_node;
        if(left_eta < r *expected_nodes && arg.large == false){
            n_exist_rr = 0;
            clear();
            if(IC) r-= r/2;
            if(LT & r > 0.2) r-= 0.1;
            else if(LT & r <= 0.2) r -= r/2;
        }
        if(left_eta < 0.005 * expected_nodes && arg.large == false){
            arg.large = true;
            arg.reuse = false;
            configure_large_block();
        }
        if (left_eta <= arg.batch)
        {
            RandBatch(left_eta);
            active_node+=left_eta;
            break;
        }
        GpuTimer gpu_timer;
        gpu_timer.Start();
        const double delta = 1. / left_n;															
        const double epsilon_prime = (arg.epsilon - delta) / (1 - delta);					
        AdaptiveSelect(arg, factor, epsilon_prime, delta);
        rr_round +=1;
        gpu_timer.Stop();
    }
    gpu_timer.Stop();
    std::cout << "List of " << seedSet.size() << " nodes in the seed set:";  
    float total_cost = 0.0;
    for(auto &a:seedSet){
        std::cout << " " << a;
        total_cost += costs[a];
    }
    std::cout << std::endl;
    std::cout << "total generate time: " << build_time << "s" << std::endl;
    std::cout << "total update time: " << revise_time << "s" << std::endl;
    std::cout << "total select time: " << select_time << "s" << std::endl;
    std::cout << "Singletime: " << gpu_timer.Elapsed()/1000 << "s" << std::endl;
    std::cout << "Singlecost: " << total_cost << std::endl;
    std::cout << "Singlespread: " << active_node << std::endl;
}

void GAAS::AdaptiveSelect(Argument & arg, const double factor, const double epsilon, const double delta)
{			
    const unsigned int batch = arg.batch;
    const double alpha = sqrt(log(6.0 / delta));
    const double beta = sqrt((Math::logcnk(left_n, batch) + log(6.0 / delta)) / factor);
    const double theta_o = 2 * (alpha + beta) * (alpha + beta);
    const double theta_max = 2 * left_n * (alpha + beta) * (alpha + beta) / epsilon / epsilon / arg.batch;
    const double i_max = ceil(log(left_n / arg.batch / epsilon / epsilon) / log(2)) + 1;
    const double a1 = log(3 * i_max / delta) + Math::logcnk(left_n, batch);			
    const double a2 = log(3 * i_max / delta);

    uint32_t last_sample = 0;
    uint32_t sample = theta_o;
    if(arg.reuse)
    {
        if (n_exist_rr > 0){
            std::vector<uint32_t> batch_set;
            GpuTimer gpu_timer_revise;
            gpu_timer_revise.Start();
            update_mrr_sets(n_exist_rr, arg.beta);
            gpu_timer_revise.Stop();
            revise_time += gpu_timer_revise.Elapsed()/1000;
            GpuTimer gpu_timer_select;
            gpu_timer_select.Start();
            float influence = batch_seed_selection(n_exist_rr, batch, batch_set);	
            gpu_timer_select.Stop(); 
            select_time += gpu_timer_select.Elapsed()/1000;
            float lower = (sqrt(influence + 2. * a1 / 9.) - sqrt(a1 / 2.))*(sqrt(influence + 2. * a1 / 9.) - sqrt(a1 / 2.)) - a1 / 18;
            float upper = (sqrt(influence / factor + a2 / 2.) + sqrt(a2 / 2.))*(sqrt(influence / factor + a2 / 2.) + sqrt(a2 / 2.));				
            float ratio = lower / upper;
            if (ratio > factor*(1 - epsilon))				
            {
                realization(batch_set, active_node);
                discover_polluted_mrr_sets<<<N_BLOCKS,32>>>(
                    thrust::raw_pointer_cast(d_active_set.data()),
                    thrust::raw_pointer_cast(d_covered_flags.data()),
                    thrust::raw_pointer_cast(d_covered_flags_num.data()),
                    thrust::raw_pointer_cast(d_visited_cnt_temp.data()),
                    n_exist_rr,
                    length,
                    thrust::raw_pointer_cast(d_rr_sets.data()),
                    thrust::raw_pointer_cast(d_rr_start.data()),
                    thrust::raw_pointer_cast(d_rr_block_end.data())
                );
                if(arg.beta){
                    thrust::sequence(d_covered_indices.begin(), d_covered_indices.end());
                    thrust::sort_by_key(d_covered_flags_num.begin(), d_covered_flags_num.end(), d_covered_indices.begin());
                }
                for (auto &a:batch_set){
                    seedSet.push_back(a);
                }
                uint32_t add_root = d_n_add_root[0];
                uint32_t regen_rr = d_n_regen_rr[0];
                std::cout << " updated RR sets: " << n_exist_rr << " clean: " << add_root << " polluted: " << regen_rr << " round time: " << (gpu_timer_revise.Elapsed()/1000 + gpu_timer_select.Elapsed()/1000)<< std::endl;
                cudaMemset(thrust::raw_pointer_cast(d_n_add_root.data()), 0, d_n_add_root.size() * sizeof(uint32_t));
                cudaMemset(thrust::raw_pointer_cast(d_n_regen_rr.data()), 0, d_n_regen_rr.size() * sizeof(uint32_t));
                d_visited_cnt = d_visited_cnt_temp;
                return;
            }
            else
            {
                sample = n_exist_rr * 2;
                last_sample = n_exist_rr;
            }
        }

        GpuTimer gpu_timer_round;
        gpu_timer_round.Start();
        while (sample < theta_max)
        {
            GpuTimer gpu_timer_build;
            gpu_timer_build.Start();
            const int n_generated_rr = gen_mrr_sets(last_sample, sample, arg.large);
            gpu_timer_build.Stop(); 
            build_time += gpu_timer_build.Elapsed()/1000;
            std::vector<uint32_t> batch_set;	
            GpuTimer gpu_timer_select;
            gpu_timer_select.Start();
            float influence = batch_seed_selection(sample, batch, batch_set);	
            gpu_timer_select.Stop(); 
            select_time += gpu_timer_select.Elapsed()/1000;
            float lower = (sqrt(influence + 2. * a1 / 9.) - sqrt(a1 / 2.))*(sqrt(influence + 2. * a1 / 9.) - sqrt(a1 / 2.)) - a1 / 18;
            float upper = (sqrt(influence / factor + a2 / 2.) + sqrt(a2 / 2.))*(sqrt(influence / factor + a2 / 2.) + sqrt(a2 / 2.));				
            float ratio = lower / upper;
            if (ratio > factor*(1 - epsilon))				
            {
                realization(batch_set, active_node);
                discover_polluted_mrr_sets<<<N_BLOCKS,32>>>(
                    thrust::raw_pointer_cast(d_active_set.data()),
                    thrust::raw_pointer_cast(d_covered_flags.data()),
                    thrust::raw_pointer_cast(d_covered_flags_num.data()),
                    thrust::raw_pointer_cast(d_visited_cnt_temp.data()),
                    sample,
                    length,
                    thrust::raw_pointer_cast(d_rr_sets.data()),
                    thrust::raw_pointer_cast(d_rr_start.data()),
                    thrust::raw_pointer_cast(d_rr_block_end.data())
                );
                cudaDeviceSynchronize(); 
                if(arg.beta){
                    thrust::sequence(d_covered_indices.begin(), d_covered_indices.end());
                    thrust::sort_by_key(d_covered_flags_num.begin(), d_covered_flags_num.end(), d_covered_indices.begin());
                }
                for (auto &a:batch_set){
                    seedSet.push_back(a);
                }
                uint32_t add_root = d_n_add_root[0];
                uint32_t regen_rr = d_n_regen_rr[0];
                gpu_timer_round.Stop();
                std::cout << " generated RR sets: " << n_generated_rr << " round time: " << gpu_timer_round.Elapsed()/1000 << std::endl;
                cudaMemset(thrust::raw_pointer_cast(d_n_add_root.data()), 0, d_n_add_root.size() * sizeof(uint32_t));
                cudaMemset(thrust::raw_pointer_cast(d_n_regen_rr.data()), 0, d_n_regen_rr.size() * sizeof(uint32_t));
                n_exist_rr = sample;
                d_visited_cnt = d_visited_cnt_temp;
                cudaMemset(thrust::raw_pointer_cast(d_n_rr_sets.data()), 0, d_n_rr_sets.size() * sizeof(uint32_t));
                return;
            }
            last_sample = sample;
            sample = sample *2;
            batch_set.clear();
        }
    }
    
    // Only regenerate mRR-sets across rounds (GAAS_gen)
    else
    {
        while (sample < theta_max)
        {
            GpuTimer gpu_timer_build;
            gpu_timer_build.Start();
            const int n_generated_rr = gen_mrr_sets(last_sample, sample, arg.large);
            gpu_timer_build.Stop(); 
            build_time += gpu_timer_build.Elapsed()/1000;
            std::vector<uint32_t> batch_set;	
            GpuTimer gpu_timer_select;
            gpu_timer_select.Start();
            float influence = batch_seed_selection(sample, batch, batch_set);	
            gpu_timer_select.Stop(); 
            select_time += gpu_timer_select.Elapsed()/1000;
            float lower = (sqrt(influence + 2. * a1 / 9.) - sqrt(a1 / 2.))*(sqrt(influence + 2. * a1 / 9.) - sqrt(a1 / 2.)) - a1 / 18;
            float upper = (sqrt(influence / factor + a2 / 2.) + sqrt(a2 / 2.))*(sqrt(influence / factor + a2 / 2.) + sqrt(a2 / 2.));				
            float ratio = lower / upper;
            if (ratio > factor*(1 - epsilon))				
            {
                realization(batch_set, active_node);
                for (auto &a:batch_set){
                    seedSet.push_back(a);
                }
                n_exist_rr = sample;
                clear();
                return;
            }
            last_sample = sample;
            sample = sample *2;
            batch_set.clear();
        }
    }
}

void GAAS::create_csr(std::string dataset, const bool reversed)
{
    offsets.resize(1);
    offsets[0] = 0;
    int n,m;
    int srcnode, dstnode;
    float prob;
    std::string prefix = "dataset/";
    std::string filename = prefix + dataset;
    std::cout << "Reading the input file and creating CSR representation of G: " << std::endl;

    {
        std::ifstream infile(filename);
        if (!infile.is_open()) {
            std::cerr << "Error: Could not open the file " << filename << std::endl;
            return ;
        }
        infile >> n;
        infile >> m;
        while( infile >> srcnode ){
            infile >> dstnode;
            if( srcnode == dstnode ) continue;
            if( reversed ) {std::swap(srcnode, dstnode);}
            if( offsets.size() <= srcnode+1 ){
                offsets.resize(srcnode+2, 0);
            }
            if( offsets.size() <= dstnode+1 ){
                offsets.resize(dstnode+2, 0);
            }
            offsets[srcnode+1]++;
        }
    }

    std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
    dests.resize(offsets.back());
    probs.resize(offsets.back());

    {
        std::ifstream infile(filename);
        std::vector<uint32_t> place_to_write = offsets;
        infile >> n;
        infile >> m;
        while( infile >> srcnode ){
            infile >> dstnode;
            if( srcnode == dstnode ) continue;
            if( reversed ) {std::swap(srcnode, dstnode);}
            prob = 1.0/(offsets[srcnode+1]-offsets[srcnode]);
            probs[ place_to_write[srcnode] ] = prob;
            dests[ place_to_write[srcnode] ] = dstnode;
            place_to_write[srcnode]++;
        }
    }
}

int GAAS::logcnk(int n, int k) {
    int ans = 0;
    for (int i = n - k + 1; i <= n; i++)
    {
        ans += log(i);
    }
    for (int i = 1; i <= k; i++)
    {
        ans -= log(i);
    }
    return ans;
}

void GAAS::load_costs(const std::string dataset, const int num_nodes){

    std::string cost_file;
    costs.resize(num_nodes, 1.0);
    cost_file="dataset/"+dataset+"_cost_001DEG.txt";
    std::cout << cost_file << std::endl;
    std::ifstream inFile;
    inFile.open(cost_file);
    if(!inFile)
    {
        std::cout<<"cannot open the cost file at "<<cost_file<<std::endl;
        exit(1);
    }
    float nodeCost;
    inFile.seekg(0, std::ios_base::beg);
    for(size_t i=0;i<num_nodes;i++)
    {
        inFile>>nodeCost;
        costs[i]=nodeCost;
    }
    inFile.close();
}

void GAAS::load_realization(int index, int dataset_No, bool IC)
{	
    for (unsigned int i = 0; i < PO.size(); ++i)PO[i].clear();
    std::vector<std::string> dataset={"DBLP_sym","Youtube_sym","flickr","livejournal"};
    std::string pw_path;
    if(IC){
        pw_path="realization/" + dataset[dataset_No] + "_pw_ic" + std::to_string(index) + ".txt";
    }
    else{
        pw_path="realization/" + dataset[dataset_No] + "_pw_lt" + std::to_string(index) + ".txt";
    }

    std::cout << pw_path <<std::endl;
    std::ifstream load_pw;
    load_pw.open(pw_path);
    uint32_t i, nbr;
    while(!load_pw.eof())
    {
        load_pw>>i>>nbr;
        PO[i].push_back(nbr);
    }
    PO[i].pop_back();
}

void GAAS::realization(std::vector<uint32_t> batch_set, int & active_node)
{
    std::deque<int> q;
    q.clear();
    unsigned int counter = 0;
    for (auto seed_node : batch_set)
    {		
        ++counter;
        active_set[seed_node] = true;
        q.push_back(seed_node);
    }

    while (!q.empty())
    {
        int expand = q.front();
        q.pop_front();
        for (auto v : PO[expand])
        {
            if (active_set[v])continue;
            q.push_back(v);
            ++counter;
            active_set[v] = true;
        }
    }
    
    active_node += counter;
    std::cout << "number of activated nodes: " << active_node;
    if (counter == 0){
        std:: cout << "No more active nodes can be activated." << std::endl;
        n_exist_rr = 0;
    }
    d_active_set = active_set;
}

void GAAS::RandBatch(int eta)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, n_nodes - 1);
	int node = dis(gen);
	while (eta > 0)
	{		
		while (active_set[node]){
            node = dis(gen);
        }
        active_set[node] = true;
		seedSet.push_back(node);
		--eta;
	}
}