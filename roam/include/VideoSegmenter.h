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
#include <fstream>

#include <opencv2/core.hpp>

#include "Configuration.h"
#include "EnergyTerms.h"
#include "Node.h"
#include "ClosedContour.h"
#include "SnapcutTerms.h"
#include "GreenTheorem.h"
#include "StarGraph.h"
#include "ContourWarper.h"
#include "GC_Energy.h"
#include "Reparametrization.h"

#include "../tools/om_utils/include/roam/utils/timer.h"

#include <omp.h>

namespace ROAM
{

/*!
 * \brief The VideoSegmenter class
 */
// -----------------------------------------------------------------------------------
class VideoSegmenter
// -----------------------------------------------------------------------------------
{
public:

    /*!
     * \brief The WarperType enum
     */
    // -------------------------------------------------------------------------------
    enum WarpType
    // -------------------------------------------------------------------------------
    {
        WARP_TRANSLATION = 101,
        WARP_SIMILARITY
    };

    /*!
     * \brief The Params struct
     */
    // -------------------------------------------------------------------------------
    struct Params
    // -------------------------------------------------------------------------------
    {
        // ---------------------------------------------------------------------------
        explicit Params(const label label_space_side = 5,
                        const FLOAT_TYPE gaussian_smoothing_factor = 1.0)
        // ---------------------------------------------------------------------------
        {
            this->label_space_side = label_space_side;
            this->gaussian_smoothing_factor = gaussian_smoothing_factor;

            this->use_gradients_unary = true;
            this->gradients_weight = 1.0f;
            this->grad_kernel_size = 3;
            this->grad_type = GradientUnary::SOBEL;

            this->use_gradient_pairwise = false;
            this->gradient_pairwise_weight = 1.0f;

            this->use_norm_pairwise = true;
            this->norm_type = NormPairwise::L2;
            this->norm_weight = 1.0f;

            this->use_temp_norm_pairwise = false;
            this->temp_norm_weight = 0.0001f;

            this->use_snapcut_pairwise = true;
            this->snapcut_weight = 100.f;
            this->snapcut_sigma_color = 0.1f;
            this->snapcut_region_height = 15.f;
            this->snapcut_number_clusters = 3;

            this->use_landmarks = false;
            this->landmark_max_area_overlap = 10.f;
            this->landmark_min_area = 25.f;
            this->landmark_min_response = 0.1f;
            this->max_number_landmarks = -1;
            this->alternative_optimization_iterations = 1;
            this->landmark_pairwise_weight = 0.1f;
            this->landmark_to_node_weight = 0.5f;
            this->landmark_to_node_radius = 10.0f;
            this->landmarks_searchspace_side = 35;

            this->warper_type = WARP_SIMILARITY;

            this->use_green_theorem_term = false;
            this->green_theorem_weight = 0.0f;
            this->use_graphcut_term = false;
            this->reparametrization_failsafe = 0.05f;

            this->temp_angle_weight = 0.0f;
        }


        void Read(cv::FileStorage & /*fs*/);
        void Write(cv::FileStorage & /*fs*/) const;
        void Print() const;

        // Node params
        label label_space_side;

        // GradUnary params
        bool use_gradients_unary;
        FLOAT_TYPE gaussian_smoothing_factor;
        FLOAT_TYPE gradients_weight;
        int grad_kernel_size;
        GradientUnary::GradType grad_type;

        // GradPairwise
        bool use_gradient_pairwise;
        FLOAT_TYPE gradient_pairwise_weight;

        // Norm Pairwise params
        bool use_norm_pairwise;
        NormPairwise::NormType norm_type;
        FLOAT_TYPE norm_weight;

        // Norm Pairwise params
        bool use_temp_norm_pairwise;
        FLOAT_TYPE temp_norm_weight;
        FLOAT_TYPE temp_angle_weight;

        // Snapcut Pairwise term
        bool use_snapcut_pairwise;
        FLOAT_TYPE snapcut_weight;
        FLOAT_TYPE snapcut_sigma_color;
        FLOAT_TYPE snapcut_region_height;
        int snapcut_number_clusters;

        // Landmarks Term
        bool use_landmarks;
        int alternative_optimization_iterations;
        FLOAT_TYPE landmark_min_response;
        FLOAT_TYPE landmark_max_area_overlap;
        FLOAT_TYPE landmark_min_area;
        FLOAT_TYPE landmark_pairwise_weight;
        FLOAT_TYPE landmark_to_node_weight;
        FLOAT_TYPE landmark_to_node_radius;
        int max_number_landmarks;
        int landmarks_searchspace_side;

        // Warper Type
        WarpType warper_type;

        // Green Theorem
        bool use_green_theorem_term;
        FLOAT_TYPE green_theorem_weight;

        // GraphCut Term
        bool use_graphcut_term;
        FLOAT_TYPE reparametrization_failsafe;

    };

    // -------------------------------------------------------------------------------
    struct ContourElementsHelper
    // -------------------------------------------------------------------------------
    {
        std::list<cv::Ptr<ROAM::DistanceUnary> > distance_unaries_per_contour_node;
        std::list<cv::Ptr<ROAM::SnapcutPairwise> >  snapcut_pairwises_per_contour_node;
        std::list<cv::Ptr<ROAM::TempNormPairwise> > tempnorm_pairwises_per_contour_node;
        std::list<cv::Ptr<ROAM::TempAnglePairwise> > tempangle_pairwises_per_contour_node;
        std::list<cv::Ptr<ROAM::GreenTheoremPairwise> > green_theorem_pairwises;
        std::list<FLOAT_TYPE> prev_norms;
        std::list<FLOAT_TYPE> prev_angles;
        cv::Ptr<ROAM::NormPairwise> contour_norm_pairwise;
        cv::Ptr<ROAM::GradientUnary> contour_gradient_unary;
        cv::Ptr<ROAM::GradientDTUnary> contour_gradient_dt_unary;
        cv::Ptr<ROAM::GenericPairwise> contour_generic_pairwise;

        void PruneContourElements(const std::vector<bool> &remove, bool use_du, bool use_sc,
                                  bool use_tn, bool use_gt);
    };

    explicit VideoSegmenter(const Params& params = Params());

    void SetContours(const std::vector<cv::Point> &contour_pts);
    void SetNextImageAuto(const cv::Mat& next_image);
    void SetNextImage(const cv::Mat& next_image);
    void SetPrevImage(const cv::Mat& prev_image);
    void SetParameters(const Params& params);

    void Write(cv::FileStorage &fs) const;
    bool IsInit() const;
    void WriteOutput(const std::string& foldername);

    std::vector<cv::Point> ProcessFrame();
    cv::Mat WritingOperations();
	void WriteTxt(const std::string &filename) const;

    VideoSegmenter::Params getParams() const;
    void setParams(const VideoSegmenter::Params &value);

private:
    bool contour_init;

    VideoSegmenter::Params params;

    cv::Mat next_image;
    cv::Mat prev_image;

    std::shared_ptr<ClosedContour> contour;
    std::shared_ptr<StarGraph> landmarks_tree;
    std::shared_ptr<ContourWarper> contour_warper;

    ContourElementsHelper contour_elements;

    ROAM::GC_Energy graphcut;
    ROAM::GlobalModel fb_model;

    bool write_masks;
    int frame_counter;
    std::string namefolder;
    cv::Mat frame_mask;
    ROAM::Timer chrono_timer_per_frame;

    FLOAT_TYPE current_contour_cost;

    cv::Mat global_foreground_likelihood;
    cv::Mat global_background_likelihood;
    cv::Mat integral_negative_ratio_foreground_background_likelihood;

    std::vector<FLOAT_TYPE> costs_per_frame;

protected:
    void performIntermediateContourMove(const std::vector<cv::Point>& move_to_point) const;
    std::vector<cv::Point> findDiffsContourMove(const std::vector<cv::Point> &move_to_point) const;
    void reinitializeNode(Node &node, const Node &next_node, int prev_node, int node_idx, ContourElementsHelper &ce) const;
    void automaticReparametrization();
    cv::Mat accumulate(const cv::Mat &input) const;
    void updateIntegralNegRatioForGreenTheorem(const cv::Mat &mask);
};

}// namespace roam
