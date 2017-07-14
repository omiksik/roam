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

#include "Configuration.h"
#include "Node.h"

#include <opencv2/core.hpp>

#include <opencv2/tracking.hpp>
#include <opencv2/video.hpp>

namespace ROAM
{

/*!
 * \brief The ContourWarper base class
 */
// -----------------------------------------------------------------------------------
class ContourWarper
// -----------------------------------------------------------------------------------
{
public:
    virtual ~ContourWarper() {}

    /*!
     * \brief Warp warps an input contour according to the estimated model
     * \param contour_input
     * \param contour_output
     */
    virtual void Warp(const std::vector<cv::Point>& contour_input,
                      std::vector<cv::Point>& contour_output) = 0;


    /*!
     * \brief Warp overloaded function for convencience
     * \param contour_input
     * \return the warped contour
     */
    virtual std::vector<cv::Point> Warp(const std::list<Node> &contour_input) = 0;
};

/*!
 * \brief Implementation of a Rigid Transform contour warper. It works by finding
 * a robust linear ransformation between sets of matched points.
 */
// -----------------------------------------------------------------------------------
class RigidTransform_ContourWarper : public ContourWarper
// -----------------------------------------------------------------------------------
{
public:
    // -------------------------------------------------------------------------------
    enum RigidTransformType
    // -------------------------------------------------------------------------------
    {
        TRANSLATION = 101,
        SIMILARITY,
        AFFINE
    };

    /*!
     * \brief The Params struct
     */
    // -------------------------------------------------------------------------------
    struct Params
    // -------------------------------------------------------------------------------
    {
        // ---------------------------------------------------------------------------
        explicit Params(const RigidTransformType rt_type = RigidTransformType::TRANSLATION)
        // ---------------------------------------------------------------------------
        {
            this->rt_type = rt_type;
        }

        RigidTransformType rt_type;
    };

    explicit RigidTransform_ContourWarper(const RigidTransform_ContourWarper::Params &params);

    virtual void Warp(const std::vector<cv::Point>& contour_input,
                      std::vector<cv::Point>& contour_output) override;

    virtual std::vector<cv::Point> Warp(const std::list<Node> &contour_input) override;

    /*!
     * \brief Init
     * \param corresponding_points_a
     * \param corresponding_points_b
     * \param mask_input
     */
    void Init(const std::vector<cv::Point> &corresponding_points_a, const std::vector<cv::Point> &corresponding_points_b, const cv::Mat &mask_input);

private:
    cv::Mat mask_input;
    std::vector<cv::Point> corr_pts_a;
    std::vector<cv::Point> corr_pts_b;
    Params parameters;
    cv::Mat rigid_transform;
};

}// namespace roam
