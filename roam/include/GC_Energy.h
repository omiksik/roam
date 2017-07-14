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
#include <limits>
#include <memory>

#include <opencv2/core.hpp>

#include "GlobalModel.h"
#include "../../tools/maxflow/maxflow/graph.h"
typedef Graph<FLOAT_TYPE,FLOAT_TYPE,FLOAT_TYPE> GraphType;

namespace ROAM
{

/*!
 * \brief The GC_Energy class
 */
class GC_Energy
{
public:
    // ---------------------------------------------------------------------------
    enum GCClasses 
    // ---------------------------------------------------------------------------
    {
        GC_BGD = 0,     //!< an obvious background pixels
        GC_FGD = 1,     //!< an obvious foreground (object) pixel
        GC_PR_BGD = 2,  //!< a possible background pixel
        GC_PR_FGD = 3   //!< a possible foreground pixel
    };
    
    /*!
     * \brief The Params struct
     */
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

    explicit GC_Energy(const GC_Energy::Params& parameters = GC_Energy::Params());

    /*!
     * \brief Segment
     * \param image
     * \param labels
     * \param closed_contour
     * \param precomputed_contour_likelihood
     * \param global_model superseeds model if passed
     * \return
     */
    cv::Mat Segment(const cv::Mat &image, const cv::Mat &labels,
                    const std::vector<cv::Point> &closed_contour = std::vector<cv::Point>(),
                    const cv::Mat &precomputed_contour_likelihood = cv::Mat(),
                    const GlobalModel &global_model = GlobalModel(),
                    const cv::Rect &slack=cv::Rect());

    bool Initialized() const;

    GlobalModel model;

private:
    GC_Energy::Params params;
    bool is_initialized;

protected:
    bool initialize(const cv::Mat &data, const cv::Mat &labels);
    bool update(const cv::Mat &data, const cv::Mat &labels);
};

}
