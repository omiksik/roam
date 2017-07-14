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
#include <tuple>

#include <opencv2/core.hpp>

#include "Configuration.h"
#include "EnergyTerms.h"
#include "Node.h"
#include "DynamicProgramming_CPUGPU.h"

#include "../../tools/cv_tools/include/ROAM_KCF.h"

namespace ROAM
{

#ifndef GET_NODE_FROM_TUPLE
#define GET_NODE_FROM_TUPLE(X) std::get<0>(X)
#endif

#ifndef GET_RECT_FROM_TUPLE
#define GET_RECT_FROM_TUPLE(X) std::get<1>(X)
#endif

#ifndef GET_KCF_FROM_TUPLE
#define GET_KCF_FROM_TUPLE(X) std::get<2>(X)
#endif

#ifndef NO_DISPLACEMENT
#define NO_DISPLACEMENT -1.f
#endif

typedef std::tuple<std::shared_ptr<Node>,cv::Rect,std::shared_ptr<KCF> > Landmark;
/*!
 * \brief The StarGraph class
 */
// -----------------------------------------------------------------------------------
class StarGraph
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
        explicit Params(const unsigned int node_side_length = 7,
                        const FLOAT_TYPE min_response_new_landmarks = 0.1f,
                        const FLOAT_TYPE max_area_overlap_landmark = 10.f,
                        const FLOAT_TYPE min_area_landmark = 20.f,
                        const int max_number_landmarks = -1,
                        const FLOAT_TYPE weight_pairwises = 0.25f)
        // ---------------------------------------------------------------------------
        {
            this->node_side_length = node_side_length;
            this->min_response_new_landmarks = min_response_new_landmarks;
            this->max_area_overlap_landmark = max_area_overlap_landmark;
            this->min_area_landmark = min_area_landmark;
            this->max_number_landmarks = max_number_landmarks;
            this->weight_pairwises = weight_pairwises;
        }

        uint node_side_length;
        FLOAT_TYPE min_response_new_landmarks;
        FLOAT_TYPE max_area_overlap_landmark;
        FLOAT_TYPE min_area_landmark;
        int max_number_landmarks;
        FLOAT_TYPE weight_pairwises;
    };

    explicit StarGraph(const StarGraph::Params &parameters = StarGraph::Params());

    void BuildDPTable();

    FLOAT_TYPE RunDPInference();
    std::vector<label> GetCurrentSolution() const;
    void ApplyMoves();

    std::shared_ptr<DPTable> GetDPTable() const;

    std::list<Landmark> graph_nodes; //!< public member (The user will manage this as he wishes graph_nodes[0] is the root of the tree)

    void UpdateLandmarks(const cv::Mat& image, const cv::Mat& mask, bool update_root = false);
    void TrackLandmarks(const cv::Mat& next_image);
    cv::Mat DrawLandmarks(const cv::Mat& image, int rad_landmark_to_node = 1);
    cv::Mat VectorOfClosestLandmarkPoints(cv::Point contour_pt, FLOAT_TYPE radius = 10.f) const;
    std::vector<size_t> VectorOfClosestLandmarkIndexes(cv::Point contour_pt, FLOAT_TYPE radius = 10.f) const;
    FLOAT_TYPE AverageDistanceToClosestLandmarkPoints(cv::Point contour_pt, FLOAT_TYPE radius = 10.f) const;
    bool DPTableIsBuilt() const;

    std::vector<cv::Point> correspondences_a;
    std::vector<cv::Point> correspondences_b;

private:
    bool dp_table_built;
    std::shared_ptr<DPTable> dp_table;
    std::vector<label> current_solution;
    StarDP dp_solver;
    Params params;

protected:
    std::vector<cv::Rect> getMSERBoxes(const cv::Mat &img, const cv::Mat &msk) const; //!< Delivers non-redundant boxes inside the mask
    bool isBoxRedundantAndTooSmall(const cv::Rect& box, const std::vector<cv::Rect>& new_boxes = std::vector<cv::Rect>()) const;
    void addLandmarks(const cv::Mat& image, const cv::Mat& mask); //!< However, we will provide some method for Landmark management
    void removeOutsiderLandmarks(const cv::Mat& mask, const cv::Mat &image=cv::Mat()); //!< Remove landmarks that are out the mask

    std::list<cv::Ptr<ROAM::GenericUnary>> generic_unaries_per_landmark;
    std::list<cv::Ptr<ROAM::TempNormPairwise>> tempnorm_pairwises_per_landmark;
};

}// namespace roam
