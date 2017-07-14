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
#include <list>
#include <limits>
#include <memory>
#include <numeric>

#include <opencv2/core.hpp>

#include "Configuration.h"
#include "EnergyTerms.h"
#include "Node.h"
#include "DynamicProgramming_CPUGPU.h"
#include "SnapcutTerms.h"
#ifdef WITH_CUDA
#include "../cuda/roam_cuda.h"
#include "../cuda/dp_cuda.h"
#endif

namespace ROAM
{

/*!
 * \brief The ClosedContour class
 */
// -----------------------------------------------------------------------------------
class ClosedContour
// -----------------------------------------------------------------------------------
{
public:

    /*!
     * \brief The Params struct
     */
    // -------------------------------------------------------------------------------
    struct Params
    // -------------------------------------------------------------------------------
    {
        // ---------------------------------------------------------------------------
        explicit Params(const unsigned int max_number_labels = 7)
        // ---------------------------------------------------------------------------
        {
            this->max_number_labels = max_number_labels;
        }

        uint max_number_labels;
    };

    explicit ClosedContour(const ClosedContour::Params &parameters = ClosedContour::Params(), bool foreground_inside = true);

    void SetForegroundSideFlag(const bool flag);
    bool GetForegroundSideFlag() const;

    void BuildDPTable();

    FLOAT_TYPE RunDPInference();
    std::vector<label> GetCurrentSolution() const;
    void ApplyMoves();

    std::shared_ptr<DPTable> GetDPTable() const;

    void ExecuteCudaPairwises(const FLOAT_TYPE win_height, const FLOAT_TYPE dist_sigma, const FLOAT_TYPE wight_snapcut,
                              const int imrows, const int imcols);

    FLOAT_TYPE GetTotalContourCost(const FLOAT_TYPE win_height, const FLOAT_TYPE dist_sigma, const FLOAT_TYPE wight_snapcut,
                                   const int imrows, const int imcols, const bool use_sc=true) const;

    /* Generate a likelihood map from trained gmm models for each edge */
    cv::Mat GenerateLikelihoodMap(const cv::Mat &image);

    cv::Mat DrawContour(const cv::Mat &image, const cv::Mat &mask=cv::Mat(), bool draw_nodes=false) const;

    std::list<Node> contour_nodes; //!< public member (The user will manage this as he wishes)

    // We still provide some nice methods
    void PruneNodes(const std::vector<bool> &to_remove);
    size_t AddNode(const cv::Point &pt, const int prev_node = -1);
    std::list<Node>::iterator AddNode(const cv::Point &pt, std::list<Node>::iterator it);
    bool IsCounterClockWise() const;

private:
    bool dp_table_built;
    std::shared_ptr<DPTable> dp_table;
    std::vector<label> current_solution;

#ifdef WITH_CUDA
    ClosedChainDPCuda dp_solver;
#else
    ClosedChainDP dp_solver;
#endif

    Params params;
    bool foreground_inside;

    std::vector<FLOAT_TYPE> costs; //!< costs computed with cuda
protected:
    FLOAT_TYPE computeContourCudaCost(const FLOAT_TYPE win_height, const FLOAT_TYPE dist_sigma, const FLOAT_TYPE weight_snapcut,
                                      const int im_rows, const int im_cols) const;
};

}// namespace roam
