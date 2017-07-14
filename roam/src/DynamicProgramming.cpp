#include "DynamicProgramming_CPUGPU.h"
#include "../tools/om_utils/include/roam/utils/timer.h"

using namespace ROAM;

//-----------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------- DPTable ----------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
OpenChainDPTable::OpenChainDPTable(const uint16_t max_number_labels, const uint16_t number_nodes)
// -----------------------------------------------------------------------------------
{
    this->max_number_labels = max_number_labels;
    this->number_nodes = number_nodes;
}

// -----------------------------------------------------------------------------------
ClosedChainDPTable::ClosedChainDPTable(const uint16_t max_number_labels, const uint16_t number_nodes)
// -----------------------------------------------------------------------------------
{
    this->max_number_labels = max_number_labels;
    this->number_nodes = number_nodes;
}

// -----------------------------------------------------------------------------------
void OpenChainDPTable::Initialize()
// -----------------------------------------------------------------------------------
{
    unary_costs = DPTableUnaries(number_nodes,
                        std::vector<FLOAT_TYPE>(max_number_labels, std::numeric_limits<FLOAT_TYPE>::infinity()) );

    pairwise_costs = DPTablePairwises(number_nodes - 1,
                        DPTableNodePairwises(max_number_labels, std::vector<FLOAT_TYPE>(max_number_labels,
                                                                        std::numeric_limits<FLOAT_TYPE>::infinity())));
}

// -----------------------------------------------------------------------------------
void ClosedChainDPTable::Initialize()
// -----------------------------------------------------------------------------------
{
    unary_costs = DPTableUnaries(number_nodes,
                        std::vector<FLOAT_TYPE>(max_number_labels, std::numeric_limits<FLOAT_TYPE>::infinity()) );

    pairwise_costs = DPTablePairwises(number_nodes,
                        DPTableNodePairwises(max_number_labels, std::vector<FLOAT_TYPE>(max_number_labels,
                                                                            std::numeric_limits<FLOAT_TYPE>::infinity())));
}

// -----------------------------------------------------------------------------------
StarDPTable::StarDPTable(const uint16_t max_number_labels, const uint16_t number_nodes)
// -----------------------------------------------------------------------------------
{
    this->max_number_labels = max_number_labels;
    this->number_nodes = number_nodes;
}

// -----------------------------------------------------------------------------------
void StarDPTable::Initialize()
// -----------------------------------------------------------------------------------
{
    unary_costs = DPTableUnaries(number_nodes,
                        std::vector<FLOAT_TYPE>(max_number_labels, std::numeric_limits<FLOAT_TYPE>::infinity()));

    pairwise_costs = DPTablePairwises(number_nodes - 1,
                        DPTableNodePairwises(max_number_labels, std::vector<FLOAT_TYPE>(max_number_labels,
                                                                        std::numeric_limits<FLOAT_TYPE>::infinity())));
}

// -----------------------------------------------------------------------------------
TreeDPTable::TreeDPTable(const uint16_t max_number_labels, const uint16_t number_root_children)
// -----------------------------------------------------------------------------------
{
    this->number_root_children = 0;
    this->number_nodes = number_root_children + 1;
    this->max_number_labels = max_number_labels;
}

// -----------------------------------------------------------------------------------
void TreeDPTable::Initialize()
// -----------------------------------------------------------------------------------
{
    unary_costs = DPTableUnaries(number_root_children+1,
                                  std::vector<FLOAT_TYPE>(max_number_labels, std::numeric_limits<FLOAT_TYPE>::infinity()));

    pairwise_costs = DPTablePairwises(number_root_children,
                                      DPTableNodePairwises(max_number_labels, std::vector<FLOAT_TYPE>(max_number_labels,
                                                                                      std::numeric_limits<FLOAT_TYPE>::infinity())));
}

// -----------------------------------------------------------------------------------
void TreeDPTable::AddStarChild(const uint16_t children_of, const uint16_t number_nodes)
// -----------------------------------------------------------------------------------
{
    DPTableUnaries buf(number_nodes,
                       std::vector<FLOAT_TYPE>(max_number_labels, std::numeric_limits<FLOAT_TYPE>::infinity()));
    this->children_unaries.push_back(buf);

    this->parent_of_children.push_back(children_of);

    this->children_pairwises.push_back(
                DPTablePairwises(number_nodes,
                                    DPTableNodePairwises(max_number_labels, std::vector<FLOAT_TYPE>(max_number_labels,
                                        std::numeric_limits<FLOAT_TYPE>::infinity())))
                );
}


// -------------------------------------------------------------------------------
std::vector<label> ChainDP::Minimize(const std::shared_ptr<DPTable> &dp_table,
                                     FLOAT_TYPE &min_cost)
// -------------------------------------------------------------------------------
{
    const auto &unary_costs = dp_table->unary_costs;
    const auto &pairwise_costs = dp_table->pairwise_costs;

    const auto n_nodes = unary_costs.size();
    assert(n_nodes > 0);

    const auto n_labels = unary_costs[0].size(); // TODO: change (now we assume all nodes have the same number of labels)
    assert(n_nodes == (pairwise_costs.size() + 1));
    assert(n_labels == pairwise_costs[0].size());

    // accumulated costs and path indices
    prev_states = std::vector<std::vector<label> >(n_nodes, std::vector<label>(n_labels, 0)); //(should be (n_nodes - 1), for convenience, we simply skip the first col)

    // -----------------------------------------
    // initialize
    std::vector<FLOAT_TYPE> prev_costs = unary_costs[0];

    // -----------------------------------------
    // forward pass
    for(auto node = 1; node < n_nodes; ++node) // for all nodes
    {
        const auto accum_costs = prev_costs;

        #pragma omp parallel for
        for(auto state = 0; state < n_labels; ++state) // for all labels
        {
            const FLOAT_TYPE &unary = unary_costs[node][state];

            label min_idx = 0;
            FLOAT_TYPE current_min_cost = std::numeric_limits<FLOAT_TYPE>::max();

            for(auto prev_state = 0; prev_state < n_labels; ++prev_state) // for all previous labels
            {
                const FLOAT_TYPE &accumulated_cost = accum_costs[prev_state];
                const FLOAT_TYPE &pairwise = pairwise_costs[node - 1][prev_state][state];

                const FLOAT_TYPE current_cost = unary + pairwise + accumulated_cost;
                if(current_cost < current_min_cost)
                {
                    current_min_cost = current_cost;
                    min_idx = prev_state;
                }
            }

            prev_costs[state] = current_min_cost;
            prev_states[node][state] = min_idx;
        }
    }

    // -----------------------------------------
    // find the min cost state
    label min_state = static_cast<label>(std::distance(std::begin(prev_costs),
                                                       std::min_element(std::begin(prev_costs), std::end(prev_costs))));

    // optionally return also the cost
    min_cost = prev_costs[min_state];

    // -----------------------------------------
    // backward pass (path reconstruction)
    std::vector<label> path(n_nodes, 0);
    for(int node = static_cast<int>(n_nodes - 1); node >= 0; --node) // should be signed
    {
        path[node] = static_cast<label>(min_state);
        min_state = prev_states[node][min_state];
    }

    return path;
}

// -------------------------------------------------------------------------------
std::vector<label> ClosedChainDP::Minimize(const std::shared_ptr<DPTable> &dp_table,
                                     FLOAT_TYPE &min_cost)
// -------------------------------------------------------------------------------
{
    const auto &unary_costs = dp_table->unary_costs;
    const auto &pairwise_costs = dp_table->pairwise_costs;

    const auto n_nodes = unary_costs.size();
    assert(n_nodes > 0);

    const auto n_labels = unary_costs[0].size(); // TODO: change (now we assume all nodes have the same number of labels)
    assert(n_nodes == pairwise_costs.size());
    assert(n_labels == pairwise_costs[0].size());

    prev_states = std::vector<std::vector<label> >(n_nodes + 1, std::vector<label>(n_labels, 0)); //(should be (n_nodes - 1), for convenience, we simply skip the first col)

    std::vector<label> path(n_nodes, 0);
    min_cost = std::numeric_limits<FLOAT_TYPE>::infinity(); // keep track which fixed label is the best one

    for(auto fixed_label = 0; fixed_label < n_labels; ++fixed_label)
    {
        // -----------------------------------------
        // solve first node, ie fix one label, set all other as infinite costs
        std::vector<FLOAT_TYPE> prev_costs(n_labels, std::numeric_limits<FLOAT_TYPE>::infinity());
        prev_costs[fixed_label] = 0;

        // -----------------------------------------
        // solve middle part
        for(size_t node = 1; node < n_nodes; ++node)
        {
            // accumulated costs?
            const auto accum_costs = prev_costs;

            #pragma omp parallel for
            for(auto state = 0; state < n_labels; ++state)
            {
                const FLOAT_TYPE &unary = unary_costs[node][state];

                label min_idx = 0;
                FLOAT_TYPE current_min_cost = std::numeric_limits<FLOAT_TYPE>::infinity();

                for(auto prev_state = 0; prev_state < n_labels; ++prev_state)
                {
                    const FLOAT_TYPE &accumulated_cost = accum_costs[prev_state];
                    const FLOAT_TYPE &pairwise = pairwise_costs[node - 1][prev_state][state];

                    const FLOAT_TYPE current_cost = unary + pairwise + accumulated_cost;
                    if(current_cost < current_min_cost)
                    {
                        current_min_cost = current_cost;
                        min_idx = prev_state;
                    }
                }
                prev_costs[state] = current_min_cost;
                prev_states[node][state] = min_idx;
            }
        }

        // -----------------------------------------
        // handle the duplicated node separately

        {
            const auto accum_costs = prev_costs;
            const FLOAT_TYPE &unary = unary_costs[0][fixed_label]; // same one as for the first node (duplicated)

            label min_idx = 0;
            FLOAT_TYPE current_min_cost = std::numeric_limits<FLOAT_TYPE>::infinity();

            for(auto prev_state = 0; prev_state < n_labels; ++prev_state)
            {
                const FLOAT_TYPE &accumulated_cost = accum_costs[prev_state];
                const FLOAT_TYPE &pairwise = pairwise_costs[n_nodes-1][prev_state][fixed_label];

                const FLOAT_TYPE current_cost = unary + pairwise + accumulated_cost;
                if(current_cost < current_min_cost)
                {
                    current_min_cost = current_cost;
                    min_idx = prev_state;
                }
            }

            prev_costs[fixed_label] = current_min_cost; // other labels are not infinite, do not USE minimum, just the fixed label (or set others to infty)
            prev_states[n_nodes][fixed_label] = min_idx;
        }

        // -----------------------------------------
        // find the terminal state (the fixed one, all other have infinite cost)
        label min_state = prev_states[n_nodes][fixed_label]; // no need to use min, all other costs are infty
        const FLOAT_TYPE &current_min_val = prev_costs[fixed_label];

        // -----------------------------------------
        // backward pass if we achieved better energy (path reconstruction)
        if(current_min_val < min_cost)
        {
            min_cost = current_min_val;

            // reconstruct the path
            for(int node = static_cast<int>(n_nodes - 1); node >= 0; --node) // should be signed
            {
                path[node] = static_cast<label>(min_state);
                min_state = prev_states[node][min_state];
            }
        }
    }

    return path;
}



// -------------------------------------------------------------------------------
std::vector<label> ClosedChainDP1::Minimize(const std::shared_ptr<DPTable> &dp_table,
                                    FLOAT_TYPE &min_cost)
// -------------------------------------------------------------------------------
{
    const auto &unary_costs = dp_table->unary_costs;
    const auto &pairwise_costs = dp_table->pairwise_costs;

    const auto n_nodes = unary_costs.size();
    assert(n_nodes > 0);

    const auto n_labels = unary_costs[0].size(); // TODO: change (now we assume all nodes have the same number of labels)
    assert(n_nodes == pairwise_costs.size());
    assert(n_labels == pairwise_costs[0].size());

    // prev_states = std::vector<std::vector<label> >(n_nodes + 1, std::vector<label>(n_labels, 0)); //(should be (n_nodes - 1), for convenience, we simply skip the first col)
    std::vector<std::vector<std::vector<label> > > prev_states(n_labels, std::vector<std::vector<label> >(n_nodes + 1, std::vector<label>(n_labels, 0))); //(should be (n_nodes - 1), for convenience, we simply skip the first col)
    std::vector<std::vector<FLOAT_TYPE> > prev_costs(n_labels, std::vector<FLOAT_TYPE>(n_labels, std::numeric_limits<FLOAT_TYPE>::infinity()));

    // -----------------------------------------
    // solve first node, ie fix one label, set all other as infinite costs
    #pragma omp parallel for
    for(auto l = 0; l < n_labels; ++l)
        prev_costs[l][l] = 0;


    // now we'll go node by node and solve everything else in parallel
    for(size_t node = 1; node < n_nodes; ++node)
    {
        // -----------------------------------------
        // for each fixed label

        //////// ONDRA: parallelize from here
        #pragma omp parallel for
        for(auto fixed_label = 0; fixed_label < n_labels; ++fixed_label)
        {
            // accumulated costs?
            const auto &accum_costs = prev_costs[fixed_label];

            for(auto state = 0; state < n_labels; ++state)
            {
                const FLOAT_TYPE &unary = unary_costs[node][state];

                label min_idx = 0;
                FLOAT_TYPE current_min_cost = std::numeric_limits<FLOAT_TYPE>::infinity();

                for(auto prev_state = 0; prev_state < n_labels; ++prev_state)
                {
                    const FLOAT_TYPE &accumulated_cost = accum_costs[prev_state];
                    const FLOAT_TYPE &pairwise = pairwise_costs[node - 1][prev_state][state];

                    const FLOAT_TYPE current_cost = unary + pairwise + accumulated_cost;
                    if(current_cost < current_min_cost)
                    {
                        current_min_cost = current_cost;
                        min_idx = prev_state;
                    }
                }
                prev_costs[fixed_label][state] = current_min_cost;
                prev_states[fixed_label][node][state] = min_idx;
            }
        }
        //////// ONDRA: stop parallelism

    }

    // -----------------------------------------
    // handle the duplicated node separately
    #pragma omp parallel for
    for(auto fixed_label = 0; fixed_label < n_labels; ++fixed_label)
    {
        const auto &accum_costs = prev_costs[fixed_label];
        const FLOAT_TYPE &unary = unary_costs[0][fixed_label]; // same one as for the first node (duplicated)

        label min_idx = 0;
        FLOAT_TYPE current_min_cost = std::numeric_limits<FLOAT_TYPE>::infinity();

        for(auto prev_state = 0; prev_state < n_labels; ++prev_state)
        {
            const FLOAT_TYPE &accumulated_cost = accum_costs[prev_state];
            const FLOAT_TYPE &pairwise = pairwise_costs[n_nodes - 1][prev_state][fixed_label];

            const FLOAT_TYPE current_cost = unary + pairwise + accumulated_cost;
            if(current_cost < current_min_cost)
            {
                current_min_cost = current_cost;
                min_idx = prev_state;
            }
        }

        prev_costs[fixed_label][fixed_label] = current_min_cost; // other labels are not infinite, do not USE minimum, just the fixed label (or set others to infty)
        prev_states[fixed_label][n_nodes][fixed_label] = min_idx;
    }

    // -----------------------------------------
    // find the terminal state (only over the fixed ones, all others have infinite cost)
    label min_state = 0;
    min_cost = std::numeric_limits<FLOAT_TYPE>::infinity();

    for(auto fixed_label = 0; fixed_label < n_labels; ++fixed_label)
    {
        const auto &cost = prev_costs[fixed_label][fixed_label];
        if(cost < min_cost)
        {
            min_cost = cost;
            min_state = fixed_label;
        }
    }

    // -----------------------------------------
    // backward pass (path reconstruction)
    const label min_fixed_label = min_state;

    min_state = prev_states[min_fixed_label][n_nodes][min_state];

    std::vector<label> path(n_nodes, 0);

    for(int node = static_cast<int>(n_nodes - 1); node >= 0; --node) // should be signed
    {
        path[node] = static_cast<label>(min_state);
        min_state = prev_states[min_fixed_label][node][min_state];
    }

    return path;
}

// -------------------------------------------------------------------------------
std::vector<label> StarDP::Minimize(const std::shared_ptr<DPTable> &dp_table,
                                        FLOAT_TYPE &min_cost)
// -------------------------------------------------------------------------------
{
    const auto &unary_costs = dp_table->unary_costs;
    const auto &pairwise_costs = dp_table->pairwise_costs;

    const auto n_nodes = unary_costs.size();
    assert(n_nodes > 0);

    const auto n_labels = unary_costs[0].size(); // TODO: change (now we assume all nodes have the same number of labels)
    assert(n_nodes == pairwise_costs.size() + 1);
    assert(n_labels == pairwise_costs[0].size());

    prev_states = std::vector<std::vector<label> >(n_nodes, std::vector<label>(n_labels, 0)); //(should be (n_nodes - 1), for convenience, we simply skip the first col)
    std::vector<FLOAT_TYPE> total_costs(n_labels, std::numeric_limits<FLOAT_TYPE>::infinity());

    // -----------------------------------------
    // forward pass
    #pragma omp parallel for
    for(auto state = 0; state < n_labels; ++state) // for all labels
    {
        FLOAT_TYPE total_cost = unary_costs[0][state]; // assuming the root node is always 0

        for(auto leaf = 1; leaf < n_nodes; ++leaf) // for all leaves
        {
            label min_idx = 0;
            FLOAT_TYPE min_cost_ij = std::numeric_limits<FLOAT_TYPE>::max();

            for(auto prev_state = 0; prev_state < n_labels; ++prev_state) // and all labels
            {
                const FLOAT_TYPE &leaf_unary = unary_costs[leaf][prev_state];
                const FLOAT_TYPE &pairwise = pairwise_costs[leaf - 1][prev_state][state];

                const FLOAT_TYPE current_cost = pairwise + leaf_unary;
                if(current_cost < min_cost_ij)
                {
                    min_cost_ij = current_cost;
                    min_idx = prev_state;
                }
            }

            // sum it up
            total_cost += min_cost_ij;

            // save path for branch
            prev_states[leaf][state] = min_idx;
        }

        total_costs[state] = total_cost;
    }


    // -----------------------------------------
    // find the min cost state
    const label min_state = static_cast<label>(std::distance(std::begin(total_costs),
                                                             std::min_element(std::begin(total_costs), std::end(total_costs))));

    // optionally return also the cost
    min_cost = total_costs[min_state];

    // -----------------------------------------
    // backward pass (path reconstruction)
    std::vector<label> tree_path(n_nodes, 0);

    // root node label
    tree_path[0] = min_state;

    #pragma omp parallel for
    for(auto leaf = 1; leaf < n_nodes; ++leaf)
        tree_path[leaf] = prev_states[leaf][min_state];

    return tree_path;
}

// -------------------------------------------------------------------------------
std::vector<label> TreeDP::Minimize(const std::shared_ptr<DPTable> &dp_table,
                                    FLOAT_TYPE &min_cost)
// -------------------------------------------------------------------------------
{
    // Tables
    const auto &dp_tree_table = std::dynamic_pointer_cast<TreeDPTable>(dp_table);

    const auto &parent_unary_costs = dp_tree_table->unary_costs;
    const auto &children_unary_costs = dp_tree_table->children_unaries;
    const auto &root_pairwise_costs = dp_tree_table->pairwise_costs;
    const auto &children_pairwise_costs = dp_tree_table->children_pairwises;

    const auto &max_number_labels = dp_tree_table->max_number_labels;

    // for each of the "children groups" of the depth-most part of the tree, maintain
    // a "shortest group path" to each one of the label of the "children group" parents

    // Notation: cg stands for children group
    const auto number_of_cg = children_pairwise_costs.size();

    // 1. START FROM THE LEAVES

    // Children group paths
    // - For each child a label is needed (the move of that child). A vector of size num_children is formed.
    // - For each possible label of the parent, one of those vectors is needed.
    // - For each children group, then, we have a vector<vector<vector>> of labels;
    std::vector<std::vector<std::vector<label> > > cg_paths(number_of_cg);
    std::vector< std::vector<FLOAT_TYPE> > cg_costs(number_of_cg);

    // For each children group
    for (size_t cg_idx=0; cg_idx<number_of_cg; ++cg_idx)
    {
        // Will be filled as a vector of moves for each parent label
        auto &cg_path = cg_paths[cg_idx];
        cg_path.resize(max_number_labels);

        auto &cg_cost = cg_costs[cg_idx];
        cg_cost.resize(max_number_labels);

        const size_t num_children = children_pairwise_costs[cg_idx].size();

        // For each label of the parent
        #pragma omp parallel for
        for (auto l_parent=0; l_parent<max_number_labels; ++l_parent)
        {
            // winner configuration cost (initialized with +inf)
            FLOAT_TYPE winner_cost = std::numeric_limits<FLOAT_TYPE>::infinity();
            std::vector<label> winner_config(num_children, label(0));

            const size_t cg_iters = static_cast<size_t>(std::pow(max_number_labels,num_children));
            for (size_t index=0; index<cg_iters; ++index)
            {

                std::vector<label> current_config;
                current_config.resize(num_children, 0);

                FLOAT_TYPE current_cost = 0;

                int d_index = static_cast<int>(index);
                size_t child_idx=0;

                while (d_index >= max_number_labels)
                {
                    const int l = d_index % max_number_labels;
                    d_index = d_index / max_number_labels;
                    current_config[child_idx] = label(l);
                    ++child_idx;
                }
                current_config[child_idx] = label(d_index);

                for (auto ix=0; ix<current_config.size(); ++ix)
                {
                    current_cost += children_unary_costs[cg_idx][ix][current_config[ix]];
                    current_cost += children_pairwise_costs[cg_idx][ix][l_parent][current_config[ix]];
                }

                if (current_cost<winner_cost)
                {
                    winner_cost = current_cost;
                    winner_config = current_config;
                }
            }

            // filled with the children moves for a given parent_label
            cg_path[l_parent] = winner_config;
            // keep track of the costs
            cg_cost[l_parent] = winner_cost;

        }
    }

    // 2. CONTINUE WITH TOP PART OF THE TREE

    std::vector<label> winner_config;
    FLOAT_TYPE winner_cost = std::numeric_limits<FLOAT_TYPE>::infinity();

    for (label l_parent=0; l_parent<max_number_labels; ++l_parent)
    {
        const size_t tg_iters = static_cast<size_t>(std::pow(max_number_labels,number_of_cg));
        #pragma omp parallel for
        for (auto index=0; index<tg_iters; ++index)
        {
            std::vector<label> current_config;
            current_config.push_back(l_parent);
            current_config.resize(number_of_cg+1, 0);

            FLOAT_TYPE current_cost = 0;

            int d_index = static_cast<int>(index);
            size_t child_idx=0;

            while (d_index >= max_number_labels)
            {
                const int l = d_index % max_number_labels;
                d_index = d_index / max_number_labels;
                current_config[child_idx+1] = label(l);
                ++child_idx;
            }
            current_config[child_idx+1] = label(d_index);

            for (auto ix=0; ix<current_config.size(); ++ix)
            {
                current_cost += parent_unary_costs[ix][current_config[ix]];

                if (ix>0)
                {
                    current_cost += root_pairwise_costs[ix-1][l_parent][current_config[ix]];
                    current_cost += cg_costs[ix-1][current_config[ix]];
                }
            }

            if (current_cost<winner_cost)
            {
                winner_cost = current_cost;
                winner_config = current_config;
            }
        }
    }

    min_cost = winner_cost;

    std::vector<label> to_append;
    for (auto ix=1; ix<winner_config.size(); ++ix)
    {
        const size_t cg=ix-1;
        const std::vector<label> &dum = cg_paths[cg][winner_config[ix]];
        to_append.insert(to_append.end(), dum.begin(), dum.end());
    }
    winner_config.insert(winner_config.end(), to_append.begin(), to_append.end());

    return winner_config;

}

// -------------------------------------------------------------------------------
std::vector<label> ClosedChainDPCuda::Minimize(const std::shared_ptr<DPTable> &dp_table,
                                    FLOAT_TYPE &min_cost)
// -------------------------------------------------------------------------------
{
    std::vector<ROAM::label> path;
#ifdef WITH_CUDA
    const auto &unary_costs = dp_table->unary_costs;
    const auto &pairwise_costs = dp_table->pairwise_costs;

    const auto n_nodes = unary_costs.size();
    assert(n_nodes > 0);

    const auto n_labels = unary_costs[0].size(); // TODO: change (now we assume all nodes have the same number of labels)
    assert(n_nodes == pairwise_costs.size());
    assert(n_labels == pairwise_costs[0].size());


    min_cost = cuda_roam::CudaDPMinimize(unary_costs, pairwise_costs, path);

#endif
    return path;
}