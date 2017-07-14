#include "GlobalModel.h"

using namespace ROAM;

// -----------------------------------------------------------------------------------
GlobalModel::GlobalModel(const GlobalModel::Params &parameters)
// -----------------------------------------------------------------------------------
{
    this->params = parameters;
    this->initialized = false;
}

// -----------------------------------------------------------------------------------
bool GlobalModel::Initialize(const cv::Mat &data, const cv::Mat &labels)
// -----------------------------------------------------------------------------------
{
    if(!this->foreground_model.initialized && !this->background_model.initialized)
    {
        foreground_model.initializeMixture(data, labels);
        background_model.initializeMixture(data, 255 - labels);

        this->initialized = (foreground_model.initialized && background_model.initialized);
        return this->initialized;
    }
    
    return false;
}

// -----------------------------------------------------------------------------------
bool GlobalModel::Update(const cv::Mat &data, const cv::Mat &labels)
// -----------------------------------------------------------------------------------
{
    if(this->initialized)
    {
        foreground_model.updateMixture(data, labels);
        background_model.updateMixture(data, 255 - labels);

        return (foreground_model.initialized && background_model.initialized);
    }
    
    return false;
}

// -----------------------------------------------------------------------------------
cv::Mat GlobalModel::ComputeLikelihood(const cv::Mat &data) const
// -----------------------------------------------------------------------------------
{
    cv::Mat likelihood[2];
    likelihood[0] = foreground_model.getLikelihood(data);
    likelihood[1] = background_model.getLikelihood(data);

    cv::Mat output;
    cv::merge(likelihood, 2, output);

    return output;
}

// -----------------------------------------------------------------------------------
cv::Vec2f GlobalModel::ComputeLikelihood(const cv::Vec3b &data) const
// -----------------------------------------------------------------------------------
{
    cv::Vec2f out( foreground_model.getLikelihood(cv::Vec3f(data)),
                   background_model.getLikelihood(cv::Vec3f(data)) );
    return out;
}

// -----------------------------------------------------------------------------------
void GlobalModel::ComputeLikelihood(const cv::Mat &data,
                                    cv::Mat &fg_likelihood,
                                    cv::Mat &bg_likelihood) const
// -----------------------------------------------------------------------------------
{
    fg_likelihood = foreground_model.getLikelihood(data);
    bg_likelihood = background_model.getLikelihood(data);
}