#pragma once

#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <regex>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

///////////////////// AUXILIARY FUNCTIONS FOR MAIN FILES /////////////////////////////

// -----------------------------------------------------------------------------------
std::vector<std::string> Split(const std::string& input, const std::string& regex);
// -----------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
std::vector<std::string> ListOfFilenames(const std::string &file_containing_list);
// -----------------------------------------------------------------------------------


// -----------------------------------------------------------------------------------
std::vector<cv::Point> ContourFromMask(const cv::Mat &mask,
                                       int type_simplification=CV_CHAIN_APPROX_NONE);
// -----------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
std::vector<std::vector<cv::Point>> ContoursFromMasks(const std::vector<cv::Mat> &masks,
                                        int type_simplification=CV_CHAIN_APPROX_NONE);
// -----------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
cv::Mat ReadImage(const std::string &filename,
                  const int opencv_read_type = cv::IMREAD_COLOR,
                  const int over_size = 0);
// -----------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
cv::Mat CreateOverlay(const cv::Mat &image, const cv::Mat &mask);
// -----------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
cv::Mat CreateWhiteBG(const cv::Mat &image, const cv::Mat &mask);
// -----------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
template<typename T>
void MatToVector(const cv::Mat& in, std::vector<T>& out)
// -----------------------------------------------------------------------------------
{
    for(int i = 0; i < in.rows; ++i)
        for(int j = 0; j < in.cols; ++j)
            out.push_back(in.at<unsigned char>(i, j));
}

