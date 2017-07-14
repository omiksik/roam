/*
This is a source code of "ROAM: a Rich Object Appearance Model with Application to Rotoscoping" implemented by Ondrej Miksik and Juan-Manuel Perez-Rua. 

@inproceedings{miksik2017roam,
  author = {Ondrej Miksik and Juan-Manuel Perez-Rua and Philip H.S. Torr and Patrick Perez},
  title = {ROAM: a Rich Object Appearance Model with Application to Rotoscoping},
  booktitle = {CVPR},
  year = {2017}
}
*/

#pragma once

#include <vector>
#include <memory>
#include <limits>
#include <algorithm>
#include <stdint.h>

#include "Configuration.h"
#include "DynamicProgramming.h"

#ifdef WITH_CUDA
#include "../cuda/dp_cuda.h"
#endif

namespace ROAM
{

/*!
 * \brief The DPTable base struct
          Assumes [node][label] and [node][l1][l2]
 */
// -----------------------------------------------------------------------------------
struct DPTable
// -----------------------------------------------------------------------------------
{
    virtual ~DPTable() {}

    DPTableUnaries unary_costs;
    DPTablePairwises pairwise_costs;

    virtual void Initialize() = 0;

   uint16_t max_number_labels;
   uint16_t number_nodes;
};


/*!
 * \brief The DynamicProgramming class
 */
// -----------------------------------------------------------------------------------
class DynamicProgramming
// -----------------------------------------------------------------------------------
{
public:
    DynamicProgramming() {}
    explicit DynamicProgramming(const uint16_t max_number_labels);
    virtual ~DynamicProgramming() {}

    virtual std::vector<label> Minimize(const std::shared_ptr<DPTable>& dp_table,
                                        FLOAT_TYPE &min_cost) = 0;

protected:
    std::vector<std::vector<label> > prev_states;
};


/*!
 * \brief The OpenChainDPTable struct
 */
// -----------------------------------------------------------------------------------
struct OpenChainDPTable : public DPTable
// -----------------------------------------------------------------------------------
{
    OpenChainDPTable(const uint16_t max_number_labels, const uint16_t number_nodes);
    void Initialize() override;
};

/*!
 * \brief The ClosedChainDPTable struct
 */
// -----------------------------------------------------------------------------------
struct ClosedChainDPTable : public DPTable
// -----------------------------------------------------------------------------------
{
    ClosedChainDPTable(const uint16_t max_number_labels, const uint16_t number_nodes);
    void Initialize() override;
};

/*!
 * \brief The StarDPTable struct
 */
// -----------------------------------------------------------------------------------
struct StarDPTable : public DPTable
// -----------------------------------------------------------------------------------
{
    StarDPTable(const uint16_t max_number_labels, const uint16_t number_nodes /*! Including root node */);
    void Initialize() override;
};

/*!
 * \brief The TreeDPTable struct
 */
// -----------------------------------------------------------------------------------
struct TreeDPTable : public DPTable
// -----------------------------------------------------------------------------------
{
    DPTableStarPairwises children_pairwises;
    DPTableStarUnaries children_unaries;

    std::vector<uint16_t> parent_of_children;

    TreeDPTable(const uint16_t max_number_labels, const uint16_t number_root_children);

    void Initialize() override;

    void AddStarChild(const uint16_t children_of, const uint16_t number_nodes);

protected:
    uint16_t number_root_children;
};


/*!
 * \brief The Chain DynamicProgramming class
 */
// -----------------------------------------------------------------------------------
class ChainDP : public DynamicProgramming
// -----------------------------------------------------------------------------------
{
public:
    // -------------------------------------------------------------------------------
    virtual std::vector<label> Minimize(const std::shared_ptr<DPTable> &dp_table,
                                        FLOAT_TYPE &min_cost)  override;
    // -------------------------------------------------------------------------------
};

/*!
 * \brief The Closed Chain DynamicProgramming class
 */
// -----------------------------------------------------------------------------------
class ClosedChainDP : public DynamicProgramming
// -----------------------------------------------------------------------------------
{
public:
    // -------------------------------------------------------------------------------
    virtual std::vector<label> Minimize(const std::shared_ptr<DPTable> &dp_table,
                                         FLOAT_TYPE &min_cost) override;
    // -------------------------------------------------------------------------------
};

/*!
* \brief The Closed Chain DynamicProgramming class
*/
// -----------------------------------------------------------------------------------
class ClosedChainDP1 : public DynamicProgramming
// -----------------------------------------------------------------------------------
{
public:
    // -------------------------------------------------------------------------------
    virtual std::vector<label> Minimize(const std::shared_ptr<DPTable> &dp_table,
                                        FLOAT_TYPE &min_cost) override;
    // -------------------------------------------------------------------------------
};


/*!
* \brief The Closed Chain DynamicProgramming class on GPU
*/
// -----------------------------------------------------------------------------------
class ClosedChainDPCuda : public DynamicProgramming
// -----------------------------------------------------------------------------------
{
public:
    // -------------------------------------------------------------------------------
    virtual std::vector<label> Minimize(const std::shared_ptr<DPTable> &dp_table,
                                        FLOAT_TYPE &min_cost) override;
    // -------------------------------------------------------------------------------
};

/*!
 * \brief The Star DynamicProgramming class
 */
// -----------------------------------------------------------------------------------
class StarDP : DynamicProgramming
// -----------------------------------------------------------------------------------
{
public:
    // -------------------------------------------------------------------------------
    virtual std::vector<label> Minimize(const std::shared_ptr<DPTable> &dp_table,
                                        FLOAT_TYPE &min_cost) override;
    // -------------------------------------------------------------------------------
};

/*!
 * \brief The Tree DynamicProgramming class
 */
// -----------------------------------------------------------------------------------
class TreeDP : DynamicProgramming
// -----------------------------------------------------------------------------------
{
public:
    // -------------------------------------------------------------------------------
    virtual std::vector<label> Minimize(const std::shared_ptr<DPTable> &dp_table,
                                        FLOAT_TYPE &min_cost) override;
    // -------------------------------------------------------------------------------
};


}
