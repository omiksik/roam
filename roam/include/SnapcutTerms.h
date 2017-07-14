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

#include "Configuration.h"
#include "EnergyTerms.h"
#include "RotatedRect.h"

#include "../../tools/cv_tools/include/ROAM_GMM.h"
#include "../../tools/om_utils/include/roam/utils/timer.h"

#ifdef WITH_CUDA
    #include "../cuda/roam_cuda.h"
#endif

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace ROAM
{

/*!
 * \brief The SnapcutPairwise class is an implementation of PariwiseTerm
 * It is based on local temporally consistent GMMs. It is implemented in
 * both CPU and GPU, as it is a very expensive term.
 */
// -----------------------------------------------------------------------------------
class SnapcutPairwise : public PairwiseTerm
// -----------------------------------------------------------------------------------
{
public:
    // -------------------------------------------------------------------------------
    struct Params
    // -------------------------------------------------------------------------------
    {
        // ---------------------------------------------------------------------------
        explicit Params(FLOAT_TYPE weight_term=0.5f,
                        FLOAT_TYPE sigma_color=0.1f,
                        FLOAT_TYPE region_height=10.0f,
                        FLOAT_TYPE node_side_length=11.f,
                        int number_clusters=3, bool block_get_cost = false)
        // ---------------------------------------------------------------------------
        {
            this->weight_term = weight_term;
            this->sigma_color = sigma_color;
            this->region_height = region_height;
            this->node_side_length = node_side_length;
            this->number_clusters = number_clusters;
            this->block_get_cost = block_get_cost; //!> @brief This flag should be activated when using cuda
        }

        FLOAT_TYPE weight_term;

        FLOAT_TYPE sigma_color;

        FLOAT_TYPE region_height;
        FLOAT_TYPE node_side_length;
        int number_clusters;
        bool block_get_cost;
    };

    virtual bool InitializeEdge(const cv::Point &coordinates_a, const cv::Point &coordinates_b,
                                const cv::Mat &reference_image = cv::Mat(),
                                const cv::Mat &reference_mask = cv::Mat(),
                                const cv::Point &diff_a = cv::Point(0,0),
                                const cv::Point &diff_b = cv::Point(0,0)) = 0;


    virtual void GetModelElements(cv::Mat& evaluated_gmms, cv::Rect& valid_enclosing_rect, bool& is_edge_imp, bool& is_edge_init) = 0;
    virtual void GetModels(GMMModel &fg_model, GMMModel &bg_model, cv::Rect &win, bool &is_edge_imp,
                           cv::Mat &fg_eval, cv::Mat &bg_eval) = 0;

    const cv::Mat& GetMatrixA() const;
    const cv::Mat& GetMatrixB() const;

    /*!
     * \brief Constructor
     * \param parameters of SnapcutPairwise
     */
    BOILERPLATE_CODE_PAIRWISE("SNAP", SnapcutPairwise)

protected:
    Params params;
};

} // namespace roam
