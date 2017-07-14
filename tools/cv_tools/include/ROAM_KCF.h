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

#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>

#include "../../../roam/include/Configuration.h"

#ifndef M_PI
    #define M_PI           3.14159265358979323846  /* pi */
#endif

namespace ROAM
{

/*!
* \brief KCF: An object tracking class based on Kernelized Correlation Filters for object tracking.
*/
// variables with "_f" suffix are in fourier domain
// -----------------------------------------------------------------------------------
class KCF
// -----------------------------------------------------------------------------------
{
public:

    // -----------------------------------------------------------------------------------
    enum Kernels
    // -----------------------------------------------------------------------------------
    {
        KRR_LINEAR,
        KRR_GAUSSIAN,
        KRR_POLYNOMIAL
    };

    // -----------------------------------------------------------------------------------
    enum Features
    // -----------------------------------------------------------------------------------
    {
        FT_GRAY,
        FT_COLOUR
    };

    // -----------------------------------------------------------------------------------
    struct Parameters
    // -----------------------------------------------------------------------------------
    {
        Parameters()
        {
            kernel = KRR_LINEAR;
            feature_type = FT_COLOUR; // Features::FT_GRAY;

            interp_patch = 0.075f;
            interp_alpha = 0.075f;

            lambda = 1e-4f;
            target_sigma = 0.1f;//0.1;

            target_padding = 1.5f;
            detection_padding = 1.5f;
            min_displacement = cv::Size(30, 30);

            train_confidence = 0.0f;
        }

        Kernels kernel;
        Features feature_type;

        FLOAT_TYPE interp_patch;
        FLOAT_TYPE interp_alpha;
        FLOAT_TYPE train_confidence;

        FLOAT_TYPE lambda;	/// regularizer
        FLOAT_TYPE target_sigma;

        cv::Size min_displacement;
        FLOAT_TYPE target_padding;
        FLOAT_TYPE detection_padding;
    };

    // -----------------------------------------------------------------------------------
    struct Model
    // -----------------------------------------------------------------------------------
    {
        cv::Mat alpha_f;
        cv::Mat x_f;
    };

    // -----------------------------------------------------------------------------------
    struct Output
    // -----------------------------------------------------------------------------------
    {
        cv::Mat patch_translation;			/// max output defines translation
        cv::Mat patch_regression;			/// max output defines center object
        cv::Mat img_regression;
        cv::Rect orig_tracked_target;		/// original patch
        cv::Rect orig_padded_target;
        cv::Rect shifted_tracked_target;	/// added delta
        cv::Rect shifted_padded_target;
        cv::Point2i patch_center;
        cv::Point2i orig_img_center;		/// orig
        cv::Point2i shifted_img_center;		/// shifted
        FLOAT_TYPE PSR;
    };

    // -----------------------------------------------------------------------------------
    explicit KCF(const Parameters &parameters = Parameters())
        : initialized_model(false), relative_sigma(-1), confidence(std::numeric_limits<FLOAT_TYPE>::infinity()), params(parameters)
    // -----------------------------------------------------------------------------------
    {
    }

    // -----------------------------------------------------------------------------------
    // Methods
    // -----------------------------------------------------------------------------------
    cv::Rect Evaluate(const cv::Mat &img, Output &out, const cv::Rect &target = cv::Rect());
    cv::Rect Evaluate(const cv::Mat &img, const cv::Rect &target = cv::Rect());
    cv::Rect Track(const cv::Mat &img, Output &out);
    void Update(const cv::Mat &img, const cv::Rect &target, const bool padded = false);
    Output GetCurrentOutput() const;
    cv::Mat GetRegressionMap() const;
    // -----------------------------------------------------------------------------------
    static cv::Rect adjustTarget(const cv::Point2i &delta, const cv::Rect &rect);
    static cv::Rect adjustTarget(const cv::Point2i &orig_center, const cv::Point2i &max_response, const cv::Rect &rect);
    static cv::Rect adjustTarget(const cv::Point2i &orig_center, const cv::Mat &patch, const cv::Rect &rect);

    // -----------------------------------------------------------------------------------
    cv::Size getTargetSize() const;
    cv::Size getWindowSize() const;
    cv::Rect getTrackedTarget() const;
    cv::Rect getPaddedTarget() const;
    cv::Point2i getCurrentReferencePoint() const;

    /// Peak to Sidelobe Ratio (PSR) providing confidence as described in Bolme 2010
    FLOAT_TYPE computePSR(const cv::Mat& response, const int half_window = 5) const;


protected:

    void init(const cv::Rect &target);
    void train(const cv::Mat &img, const cv::Rect &target, const bool initialize = false);
    void update(const Model &current, const FLOAT_TYPE interp_patch, const FLOAT_TYPE interp_alpha);

    cv::Mat getSubwindow(const cv::Mat &img, const cv::Rect &rect) const;
    cv::Mat putSubwindow(const cv::Size &img_size, const cv::Rect &rect, const cv::Mat &subpart) const;
    cv::Mat swapTranslation2Pixels(const cv::Mat &translations) const;

    cv::Mat getFeatures(const cv::Mat &img, const Features &ft_type, const cv::Mat &window = cv::Mat()) const;
    cv::Mat getGrayFeatures(const cv::Mat &img) const;
    cv::Mat getColourFeatures(const cv::Mat &img) const;

    //cv::Mat getHannWindow(const cv::Size &sz);
    cv::Mat getHannWindow(const cv::Size &sz, const cv::Size &target_sz = cv::Size()) const;
    cv::Mat getGaussianShapedLabels(const cv::Size &sz, const FLOAT_TYPE sigma) const;

    // we use nonunitary fft, unitarity is handled separately in kernels
    cv::Mat nonunitaryFFT(const cv::Mat &mat) const;
    cv::Mat nonunitaryIFFT(const cv::Mat &mat_f) const;

    cv::Mat evaluateKernelCorrelation(const cv::Mat &a_f, const cv::Mat &b_f, const Kernels &type) const;
    cv::Mat evaluateLinearKernel(const cv::Mat &a_f, const cv::Mat &b_f) const;
    cv::Mat evaluateGaussianKernel(const cv::Mat &a_f, const cv::Mat &b_f) const;
    cv::Mat evaluatePolynomialKernel(const cv::Mat &a_f, const cv::Mat &b_f) const;

    cv::Mat specMultiplication(const cv::Mat &a_f, const cv::Mat &b_f, const bool conjB = true) const;

    Model model;			/// learnt model
    cv::Mat y_f;			/// desired output

    cv::Mat hann_window_target;
    cv::Mat hann_window_detection;

    cv::Rect tracked_target;	/// actual target
    cv::Rect padded_detection;	/// padded area (due to Hann window damping, etc.)

    bool initialized_model;
    FLOAT_TYPE relative_sigma;

    FLOAT_TYPE confidence;

    Parameters params;

    Output output;
};

}
