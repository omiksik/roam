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

#include "Configuration.h"

#include <opencv2/core.hpp>

namespace ROAM
{

/*!
 * \brief The Line class. It is used by the RotatedRect class
 */
// -----------------------------------------------------------------------------------
class Line
// -----------------------------------------------------------------------------------
{
public:
    explicit Line(const cv::Point2f &p1, const cv::Point2f &p2);
    explicit Line(const FLOAT_TYPE m = 1, const FLOAT_TYPE b = 0);

    /*!
     * \brief Y = M*X+B
     * \param x
     * \return
     */
    FLOAT_TYPE Y(const FLOAT_TYPE x) const;

    /*!
     * \brief M slope of the line
     * \return
     */
    FLOAT_TYPE M() const;

    /*!
     * \brief B
     * \return
     */
    FLOAT_TYPE B() const;

    /*!
     * \brief perpLinePassPoint returns the line that is perpendicular to input line passing by point p2.
     * \param perp_to
     * \param p1
     * \return
     */
    static Line perpLinePassPoint(const Line& perp_to, const cv::Point& p1);
protected:
    FLOAT_TYPE m;
    FLOAT_TYPE b;
};

/*!
 * \brief The LineIntegralImage class
 */
// -----------------------------------------------------------------------------------
class LineIntegralImage
// -----------------------------------------------------------------------------------
{
public:
    LineIntegralImage(){}
    static LineIntegralImage CreateFromImage(const cv::Mat& image);
    cv::Mat data;
};

/*!
 * \brief The RotatedRect class is at the heart of the SnapcutTerm
 */
// -----------------------------------------------------------------------------------
class RotatedRect
// -----------------------------------------------------------------------------------
{
public:
    RotatedRect();
    explicit RotatedRect(const cv::Point2f& po_1, const cv::Point2f& po_2, const FLOAT_TYPE height, const bool rect_up);
    explicit RotatedRect(const cv::Rect& cv_rect);

    /*!
     * \brief SumOver an image (integral)
     * \param verticalLineIntegralImage
     * \return the sum
     */
    cv::Vec3d SumOver(const LineIntegralImage &verticalLineIntegralImage) const;

    /*!
     * \brief GetCorners
     * \param pA
     * \param pB
     * \param pC
     * \param pD
     */
    void GetCorners(cv::Point2f &pA, cv::Point2f &pB, cv::Point2f &pC, cv::Point2f &pD) const;

    /*!
     * \brief GetLines
     * \param lAB
     * \param lBC
     * \param lCD
     * \param lDA
     */
    void GetLines(Line &lAB, Line &lBC, Line &lCD, Line &lDA) const;

    /*!
     * \brief GetCVRotatedRect
     * \return
     */
    cv::RotatedRect GetCVRotatedRect() const;

    /*!
     * \brief Area
     * \return
     */
    FLOAT_TYPE Area() const;

    /*!
     * \brief BuildDistanceAndColorVectors
     * \param reference_image
     * \param colors
     * \param distances_line_cd
     */
    void BuildDistanceAndColorVectors(const cv::Mat& reference_image, std::vector<cv::Vec3f>& colors,
                                      std::vector<FLOAT_TYPE>& distances_line_cd,
                                      std::vector<cv::Point> &points, bool build_points = false) const;


    /*!
     * \brief BuildDistanceAndColorVectors
     * \param reference_image
     * \param valid_mask should be == to mask_label (mask of same size of ref_img).
     * \param colors
     * \param distances_line_cd
     */
    void BuildDistanceAndColorVectors(const cv::Mat& reference_image, const cv::Mat& valid_mask, const unsigned char mask_label,
                                      std::vector<cv::Vec3f> &colors, std::vector<FLOAT_TYPE>& distances_line_cd,
                                      std::vector<cv::Point> &points,
                                      bool build_points = false) const;

    /*!
     * \brief BuildDistanceAndColorVectors_2
     * \param reference_image
     * \param colors
     * \param distances_line_cd
     * \param points
     * \param build_points
     */
    void BuildDistanceAndColorVectors_2(const cv::Mat& reference_image, std::vector<cv::Vec3f>& colors,
                                        std::vector<FLOAT_TYPE>& distances_line_cd,
                                        std::vector<cv::Point> &points, bool build_points = false) const;

    /*!
     * \brief BuildDistanceAndColorVectors_2
     * \param reference_image
     * \param valid_mask
     * \param mask_label
     * \param colors
     * \param distances_line_cd
     * \param points
     * \param build_points
     */
    void BuildDistanceAndColorVectors_2(const cv::Mat& reference_image, const cv::Mat& valid_mask, const unsigned char mask_label,
                                       std::vector<cv::Vec3f> &colors, std::vector<FLOAT_TYPE>& distances_line_cd,
                                       std::vector<cv::Point> &points,
                                       bool build_points = false) const;


private:
    cv::Point2f pA, pB, pC, pD;
    Line lAB, lBC, lCD, lDA;
    bool rect_up;
    FLOAT_TYPE half_perimeter;
    bool hack_flash;
#if defined(_MSC_VER) && (_MSC_VER <= 1800) // fix MSVC partial implementation of constexpr
    static const FLOAT_TYPE inner_rect_threshold;
#else
    static constexpr FLOAT_TYPE inner_rect_threshold = 0.01;
#endif
};

}// namespace roam
