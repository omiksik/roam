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
#include "GlobalModel.h"
#include "EnergyTerms.h"

#include "../../tools/cv_tools/include/ROAM_GMM.h"
#include "../../tools/om_utils/include/roam/utils/timer.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


namespace ROAM
{

// -----------------------------------------------------------------------------------
class GreenTheoremPairwise : public PairwiseTerm
// -----------------------------------------------------------------------------------
{
public:
    // -------------------------------------------------------------------------------
    struct Params
    // -------------------------------------------------------------------------------
    {
        // ---------------------------------------------------------------------------
        explicit Params(const FLOAT_TYPE weight_term = 0.5f, 
                        const bool contour_is_ccw = true,
                        const size_t id_a = 0, 
                        const size_t id_b = 1)
        // ---------------------------------------------------------------------------
        {
            this->weight_term = weight_term;
            this->contour_is_ccw = contour_is_ccw;
            this->id_a = id_a;
            this->id_b = id_b;
        }

        FLOAT_TYPE weight_term;
        bool contour_is_ccw;
        size_t id_a, id_b;
    };

    virtual bool InitializeEdge() = 0;

    const cv::Mat& GetMatrixA() const;
    const cv::Mat& GetMatrixB() const;

    /*!
     * \brief Constructor
     * \param parameters of SnapcutPairwise
     */
    BOILERPLATE_CODE_PAIRWISE("GREEN", GreenTheoremPairwise)

protected:
    Params params;
};

} // namespace roam
