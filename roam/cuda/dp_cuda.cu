//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include <cstdlib>

#include "dp_cuda.h"


namespace cuda_roam
{


/*!
* \brief The Closed Chain DynamicProgramming class
*/

// -----------------------------------------------------------------------------------
struct ClosedChainDPCudaSumsFunctor
// -----------------------------------------------------------------------------------
{
    const int n_labels;
    const int node;

    const thrust::device_ptr<FLOAT_TYPE> d_unary_costs;
    const thrust::device_ptr<FLOAT_TYPE> d_pairwise_costs;
    const thrust::device_ptr<FLOAT_TYPE> d_accumulated_cost;

    ClosedChainDPCudaSumsFunctor(const thrust::device_ptr<FLOAT_TYPE> unary,
                             const thrust::device_ptr<FLOAT_TYPE> pairwise,
                             const thrust::device_ptr<FLOAT_TYPE> accumulated,
                             const int _n_labels,
                             const int _node)
                             : d_unary_costs(unary), d_pairwise_costs(pairwise), d_accumulated_cost(accumulated), n_labels(_n_labels), node(_node)
    {}


    template <typename T>
    CUDA_METHOD
    void operator()(T tuple) const
    {
        // compute indices
        const int &index = thrust::get<1>(tuple);

        const int fixed_label = index / (n_labels * n_labels);
        const int state = (index - fixed_label * n_labels * n_labels) / n_labels;
        const int prev_state = index - fixed_label * n_labels * n_labels - state * n_labels;

        const int prev_cost_index = fixed_label * n_labels + prev_state;
        const int pairwise_index = prev_state * n_labels + state;

        const int node_offset = node * n_labels;
        const int pairwise_offset = (node - 1) * n_labels * n_labels;

        const FLOAT_TYPE &unary = d_unary_costs[node_offset + state];
        const FLOAT_TYPE &accumulated_cost = d_accumulated_cost[prev_cost_index];
        const FLOAT_TYPE &pairwise = d_pairwise_costs[pairwise_offset + pairwise_index]; 

        const FLOAT_TYPE current_cost = unary + pairwise + accumulated_cost;

        thrust::get<0>(tuple) = current_cost;
    }
};

// -----------------------------------------------------------------------------------
struct ClosedChainDPCudaMinsFunctor
// -----------------------------------------------------------------------------------
{
    const int n_labels;
    const int n_nodes;
    const int node;
    const thrust::device_ptr<FLOAT_TYPE> d_tmp_min_costs3;
    thrust::device_ptr<ROAM::label> d_prev_states;

    ClosedChainDPCudaMinsFunctor(const thrust::device_ptr<FLOAT_TYPE> _tmp_min_costs3,
                                 thrust::device_ptr<ROAM::label> _prev_states,
                                 const int _n_labels,
                                 const int _n_nodes,
                                 const int _node)
                             : d_tmp_min_costs3(_tmp_min_costs3), n_labels(_n_labels), n_nodes(_n_nodes), node(_node)
                             , d_prev_states(_prev_states)
    {}

    template <typename T>
    CUDA_METHOD
    void operator()(T tuple) const
    {
        // figure out indices
        const int &index = thrust::get<1>(tuple);

        const int fixed_label = index / n_labels;
        const int state = index % n_labels;

        const int idx = fixed_label * n_labels * n_labels + state * n_labels;

        int min_idx = 0;
        FLOAT_TYPE min_sum = d_tmp_min_costs3[idx];

        for(int i = 1; i < n_labels; ++i)
        {
           const FLOAT_TYPE &sum = d_tmp_min_costs3[idx + i];
           if(sum < min_sum)
           {
               min_sum = sum;
               min_idx = i;
           }
        }

        thrust::get<0>(tuple) = min_sum;
        // thrust::get<1>(tuple) = min_idx; // this has to be fixed during copying

        // [fixed_label][node][state]
        // n_labels * (n_nodes + 1) * n_labels
        const int prev_states_index = fixed_label * (n_nodes + 1) * n_labels + node * n_labels + state;
        d_prev_states[prev_states_index] = min_idx;
    }
};


// -----------------------------------------------------------------------------------
struct ClosedChainDPCudaSumsFixedFunctor
// -----------------------------------------------------------------------------------
{
    const int n_labels;
    const int n_nodes;

    const thrust::device_ptr<FLOAT_TYPE> d_unary_costs;
    const thrust::device_ptr<FLOAT_TYPE> d_pairwise_costs;
    const thrust::device_ptr<FLOAT_TYPE> d_accumulated_cost;

    ClosedChainDPCudaSumsFixedFunctor(const thrust::device_ptr<FLOAT_TYPE> unary,
                             const thrust::device_ptr<FLOAT_TYPE> pairwise,
                             const thrust::device_ptr<FLOAT_TYPE> accumulated,
                             const int _n_labels,
                             const int _n_nodes)
                             : d_unary_costs(unary), d_pairwise_costs(pairwise), d_accumulated_cost(accumulated), n_labels(_n_labels), n_nodes(_n_nodes)
    {}


    template <typename T>
    CUDA_METHOD
    void operator()(T tuple) const
    {
        // compute indices
        const int &index = thrust::get<1>(tuple);

        const int fixed_label = index / n_labels;
        const int prev_state = index % n_labels;

        const int prev_cost_index = fixed_label * n_labels + prev_state;

        const int pairwise_offset = (n_nodes - 1) * n_labels * n_labels;
        const int pairwise_index = prev_state * n_labels + fixed_label;

        const FLOAT_TYPE &unary = d_unary_costs[fixed_label]; // same one as for the first node (duplicated)
        const FLOAT_TYPE &accumulated_cost = d_accumulated_cost[prev_cost_index];
        const FLOAT_TYPE &pairwise = d_pairwise_costs[pairwise_offset + pairwise_index];

        const FLOAT_TYPE current_cost = unary + pairwise + accumulated_cost;

        thrust::get<0>(tuple) = current_cost;

        // const FLOAT_TYPE &unary = unary_costs[0][fixed_label]; // same one as for the first node (duplicated)
        // const FLOAT_TYPE &accumulated_cost = prev_costs[fixed_label][prev_state];
        // const FLOAT_TYPE &pairwise = pairwise_costs[n_nodes - 1][prev_state][fixed_label];
    }
};

// -----------------------------------------------------------------------------------
struct ClosedChainDPCudaMinsFixedFunctor
// -----------------------------------------------------------------------------------
{
    const int n_labels;
    const int n_nodes;
    const thrust::device_ptr<FLOAT_TYPE> d_tmp_min_costs2;
    thrust::device_ptr<ROAM::label> d_prev_states;

    ClosedChainDPCudaMinsFixedFunctor(const thrust::device_ptr<FLOAT_TYPE> _tmp_min_costs2,
                                 thrust::device_ptr<ROAM::label> _prev_states,
                                 const int _n_labels,
                                 const int _n_nodes)
                             : d_tmp_min_costs2(_tmp_min_costs2), n_labels(_n_labels), n_nodes(_n_nodes), 
                             d_prev_states(_prev_states)
    {}

    template <typename T>
    CUDA_METHOD
    void operator()(T tuple) const
    {
        // figure out indices
        const int &fixed_label = thrust::get<1>(tuple);

        const int idx = fixed_label * n_labels;

        int min_idx = 0;
        FLOAT_TYPE min_sum = d_tmp_min_costs2[idx];

        for(int i = 1; i < n_labels; ++i)
        {
           const FLOAT_TYPE &sum = d_tmp_min_costs2[idx + i];
           if(sum < min_sum)
           {
               min_sum = sum;
               min_idx = i;
           }
        }

        thrust::get<0>(tuple) = min_sum;

        // [fixed_label][n_nodes][fixed_label]
        // n_labels * (n_nodes + 1) * n_labels
        const int prev_states_index = fixed_label * (n_nodes + 1) * n_labels + n_nodes * n_labels + fixed_label;
        d_prev_states[prev_states_index] = min_idx; 
    }
};



// -------------------------------------------------------------------------------
FLOAT_TYPE CudaDPMinimize(
        const ROAM::DPTableUnaries &unary_costs,
        const ROAM::DPTablePairwises &pairwise_costs,
        std::vector<ROAM::label> &path
                          )
// -------------------------------------------------------------------------------
{
    FLOAT_TYPE min_cost = 0.0f;

    const size_t n_nodes = unary_costs.size();
    const size_t n_labels = unary_costs[0].size();

    // -----------------------------------------
    // N-D arrays to 1D arrays  
    std::vector<FLOAT_TYPE> unary_costs_1D(n_nodes * n_labels);
    #pragma omp parallel for
    for(auto node = 0; node < n_nodes; ++node)
    {
        const size_t offset = node * n_labels;
        std::move(unary_costs[node].begin(), unary_costs[node].end(), unary_costs_1D.begin() + offset);
    }

    std::vector<FLOAT_TYPE> pairwise_costs_1d(n_nodes * n_labels * n_labels, 0);
    #pragma omp parallel for
    for(auto node = 0; node < n_nodes; ++node)
    {
        #pragma omp parallel for
        for(auto label = 0; label < n_labels; ++label)
        {
            const size_t offset = node * n_labels * n_labels + label * n_labels;
            std::move(pairwise_costs[node][label].begin(), pairwise_costs[node][label].end(), pairwise_costs_1d.begin() + offset);
        }
    }

    // -----------------------------------------
    // device memory
    thrust::device_vector<FLOAT_TYPE> d_unary_costs(unary_costs_1D.size(), std::numeric_limits<FLOAT_TYPE>::infinity());
    thrust::device_vector<FLOAT_TYPE> d_pairwise_costs(pairwise_costs_1d.size(), std::numeric_limits<FLOAT_TYPE>::infinity());

    thrust::device_vector<FLOAT_TYPE> d_final_costs(n_labels, std::numeric_limits<FLOAT_TYPE>::infinity());
    thrust::device_vector<FLOAT_TYPE> d_prev_costs(n_labels * n_labels, std::numeric_limits<FLOAT_TYPE>::infinity());
    thrust::device_vector<ROAM::label> d_prev_states(n_labels * (n_nodes + 1) * n_labels, 0); //(should be (n_nodes - 1), for convenience, we simply skip the first col)

    thrust::device_vector<FLOAT_TYPE> d_tmp_min_costs2(n_labels * n_labels, std::numeric_limits<FLOAT_TYPE>::infinity());
    thrust::device_vector<FLOAT_TYPE> d_tmp_min_costs3(n_labels * n_labels * n_labels, std::numeric_limits<FLOAT_TYPE>::infinity());

    // counting iterators
    thrust::counting_iterator<int> init_cont_sums(0);
    thrust::counting_iterator<int> end_cont_sums = init_cont_sums + n_labels * n_labels * n_labels;

    thrust::counting_iterator<int> init_cont_mins(0);
    thrust::counting_iterator<int> end_cont_mins = init_cont_mins + n_labels * n_labels;

    thrust::counting_iterator<int> init_cont_fixed_sums(0);
    thrust::counting_iterator<int> end_cont_fixed_sums = init_cont_sums + n_labels * n_labels;

    thrust::counting_iterator<int> init_cont_fixed_mins(0);
    thrust::counting_iterator<int> end_cont_fixed_mins = init_cont_mins + n_labels;


    // -----------------------------------------
    // upload unary and pairwise from CPU to GPU
    thrust::copy(unary_costs_1D.begin(), unary_costs_1D.end(), d_unary_costs.begin());
    thrust::copy(pairwise_costs_1d.begin(), pairwise_costs_1d.end(), d_pairwise_costs.begin());

    // -----------------------------------------
    // solve first node, ie fix one label, set all other as infinite costs
    #pragma omp parallel for
    for(auto l = 0; l < n_labels; ++l)
        d_prev_costs[l * n_labels + l] = 0; // ONDRA: hope all other values were +inf?

    // now we'll go node by node and solve everything else in parallel
    for(size_t node = 1; node < n_nodes; ++node)
    {
        // compute sums
        thrust::for_each(thrust::device,
            thrust::make_zip_iterator(thrust::make_tuple(d_tmp_min_costs3.begin(), init_cont_sums)),
            thrust::make_zip_iterator(thrust::make_tuple(d_tmp_min_costs3.end(), end_cont_sums)),
            ClosedChainDPCudaSumsFunctor(d_unary_costs.data(), d_pairwise_costs.data(), d_prev_costs.data(), static_cast<int>(n_labels), static_cast<int>(node)));

        // find mins
        thrust::for_each(thrust::device,
                   thrust::make_zip_iterator(thrust::make_tuple(d_prev_costs.begin(), init_cont_mins)),
                   thrust::make_zip_iterator(thrust::make_tuple(d_prev_costs.end(), end_cont_mins)),
                   ClosedChainDPCudaMinsFunctor(d_tmp_min_costs3.data(), d_prev_states.data(), n_labels, static_cast<int>(n_nodes), static_cast<int>(node)));
    }

    // -----------------------------------------
    // handle the duplicated node separately
    {
        // sums
        thrust::for_each(thrust::device,
            thrust::make_zip_iterator(thrust::make_tuple(d_tmp_min_costs2.begin(), init_cont_fixed_sums)),
            thrust::make_zip_iterator(thrust::make_tuple(d_tmp_min_costs2.end(), end_cont_fixed_sums)),
            ClosedChainDPCudaSumsFixedFunctor(d_unary_costs.data(), d_pairwise_costs.data(), d_prev_costs.data(), static_cast<int>(n_labels), static_cast<int>(n_nodes)));

        // mins
       thrust::for_each(thrust::device,
           thrust::make_zip_iterator(thrust::make_tuple(d_final_costs.begin(), init_cont_fixed_mins)),
           thrust::make_zip_iterator(thrust::make_tuple(d_final_costs.end(), end_cont_fixed_mins)),
           ClosedChainDPCudaMinsFixedFunctor(d_tmp_min_costs2.data(), d_prev_states.data(), static_cast<int>(n_labels), static_cast<int>(n_nodes)));
    }


    // -----------------------------------------
    // find the terminal state (only over the fixed ones, all others have infinite cost)
    const ROAM::label min_fixed_label = static_cast<ROAM::label>(thrust::distance(d_final_costs.begin(), thrust::min_element(d_final_costs.begin(), d_final_costs.end())));
    min_cost = d_final_costs[min_fixed_label]; // this is our energy

    // -----------------------------------------
    // backward pass (path reconstruction)

    ROAM::label min_state = d_prev_states[n_nodes];
    thrust::device_vector<ROAM::label> d_path(n_nodes, 0);

    for(int node = static_cast<int>(n_nodes - 1); node >= 0; --node) // should be signed
    {
        d_path[node] = static_cast<ROAM::label>(min_state);

        const int idx = static_cast<int>(min_fixed_label * (n_nodes + 1) * n_labels + node * n_labels + min_state);
        min_state = d_prev_states[idx];
    }

    path.resize(d_path.size());
    thrust::copy(d_path.begin(), d_path.end(), path.begin());

    return min_cost;
}


} // namespace cuda_roam
