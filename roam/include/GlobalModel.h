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
#include "../../tools/cv_tools/include/ROAM_GMM.h"

#include <opencv2/core.hpp>


namespace ROAM
{

/*!
 * \brief The GlobalModel class
 */
// -----------------------------------------------------------------------------------
class GlobalModel
// -----------------------------------------------------------------------------------
{
public:

    struct Params
    {
        // ---------------------------------------------------------------------------
        explicit Params(const unsigned int iterations = 5, const FLOAT_TYPE betha_nd = 50.0f,
                        const FLOAT_TYPE gamma_gc = 50.0, const FLOAT_TYPE lambda_gc = 450.0f)
        // ---------------------------------------------------------------------------
        {
            this->iterations = iterations;
            this->betha_nd = betha_nd;
            this->gamma_gc = gamma_gc;
            this->lambda_gc = lambda_gc;
        }

        unsigned int iterations;
        FLOAT_TYPE betha_nd;
        FLOAT_TYPE gamma_gc;
        FLOAT_TYPE lambda_gc;
    };

    explicit GlobalModel(const GlobalModel::Params& parameters = GlobalModel::Params());

    bool Initialize(const cv::Mat &data, const cv::Mat &labels);
    bool Update(const cv::Mat &data, const cv::Mat &labels);

    cv::Mat ComputeLikelihood(const cv::Mat &data) const;
    void ComputeLikelihood(const cv::Mat &data, cv::Mat &fg_likelihood, cv::Mat &bg_likelihood) const;
    cv::Vec2f ComputeLikelihood(const cv::Vec3b &color) const;

    bool initialized;

private:
    cv::Mat data;
    cv::Mat label;
    GMMModel foreground_model;
    GMMModel background_model;
    GlobalModel::Params params;
};

}// namespace roam
