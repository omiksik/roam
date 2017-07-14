#include "ContourWarper.h"

using namespace ROAM;

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
// -----------------------------------------------------------------------------------
RigidTransform_ContourWarper::RigidTransform_ContourWarper(const RigidTransform_ContourWarper::Params &params)
// -----------------------------------------------------------------------------------
{
    this->parameters = params;
}

// -----------------------------------------------------------------------------------
void RigidTransform_ContourWarper::Init(const std::vector<cv::Point> &corresponding_points_a, 
                                        const std::vector<cv::Point> &corresponding_points_b, 
                                        const cv::Mat &mask_input)
// -----------------------------------------------------------------------------------
{
    this->corr_pts_a = corresponding_points_a;
    this->corr_pts_b = corresponding_points_b;

    if(this->corr_pts_a.size() < 5)
        this->rigid_transform = cv::Mat();

    // Apply Transformations given the corresponding points (Typically the corresponding points will come from landmark tracking
    switch(this->parameters.rt_type)
    {
        default:
        case TRANSLATION:
        {
            FLOAT_TYPE tx = 0;
            FLOAT_TYPE ty = 0;

            for(size_t p=0; p<corresponding_points_a.size(); ++p)
            {
                tx += static_cast<FLOAT_TYPE>(corresponding_points_b[p].x - corresponding_points_a[p].x);
                ty += static_cast<FLOAT_TYPE>(corresponding_points_b[p].y - corresponding_points_a[p].y);
            }
            tx /= static_cast<FLOAT_TYPE>(corresponding_points_a.size()) + std::numeric_limits<FLOAT_TYPE>::epsilon();
            ty /= static_cast<FLOAT_TYPE>(corresponding_points_b.size()) + std::numeric_limits<FLOAT_TYPE>::epsilon();

            this->rigid_transform = (cv::Mat_<FLOAT_TYPE>(2,3) << 1.f, 0.f, tx, 0.f, 1.f, ty);

            break;
        }
        case AFFINE:
        {
            this->rigid_transform = cv::estimateRigidTransform(corresponding_points_a, corresponding_points_b, true);
            break;
        }
        case SIMILARITY:
        {
            this->rigid_transform = cv::estimateRigidTransform(corresponding_points_a, corresponding_points_b, false);
            break;
        }
    }
}

// -----------------------------------------------------------------------------------
void RigidTransform_ContourWarper::Warp(const std::vector<cv::Point> &contour_input, 
                                        std::vector<cv::Point> &contour_output) // TODO: rather through return?
// -----------------------------------------------------------------------------------
{
    if(this->rigid_transform.rows > 0)
        cv::transform(contour_input, contour_output, this->rigid_transform);
    else
        contour_output = contour_input;
}

// -----------------------------------------------------------------------------------
std::vector<cv::Point> RigidTransform_ContourWarper::Warp(const std::list<Node> &contour_input)
// -----------------------------------------------------------------------------------
{
    std::vector<cv::Point> input, output;
    input.reserve(contour_input.size());
    output.reserve(contour_input.size());

    // TODO: make it running in parallel
    for(auto it_p = contour_input.begin(); it_p != contour_input.end(); ++it_p)
       input.push_back(it_p->GetCoordinates());

    if(this->rigid_transform.rows > 0)
        cv::transform(input, output, this->rigid_transform);
    else
        output = input;

    return output;
}

