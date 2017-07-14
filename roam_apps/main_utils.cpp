#include "main_utils.h"


#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <regex>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// -----------------------------------------------------------------------------------
std::vector<std::string> Split(const std::string& input, const std::string& regex)
// -----------------------------------------------------------------------------------
{
    std::regex re(regex);
    std::sregex_token_iterator
        first{input.begin(), input.end(), re, -1},
        last;
    return {first, last};
}

// -----------------------------------------------------------------------------------
std::vector<std::string> ListOfFilenames(const std::string &file_containing_list)
// -----------------------------------------------------------------------------------
{
    std::ifstream input(file_containing_list);
    std::vector<std::string> list_of_files;

    if(!input)
      return list_of_files;

    std::vector<std::string> tokens = Split(file_containing_list, "/+");
    std::string preamble;

    for (int i=0; i<tokens.size()-1; ++i)
        preamble = preamble + tokens[i] + std::string("/");

    std::string line;
    while (std::getline(input, line))
    {
        if (line.find("#")!=std::string::npos)
            continue;

        std::string to_be_pushed_to_output = preamble + line;
        list_of_files.push_back(to_be_pushed_to_output);
    }

    return list_of_files;
}

// -----------------------------------------------------------------------------------
std::vector<cv::Point> ContourFromMask(const cv::Mat &mask, int type_simplification)
// -----------------------------------------------------------------------------------
{
    assert(!mask.empty());
    std::vector<cv::Point> output_contour;

    std::vector<std::vector<cv::Point> > contours;
    cv::findContours( mask, contours, CV_RETR_EXTERNAL, type_simplification );

    // Choose largest contour
    double maxSize = 0;
    int maxIndex = 0;
    for ( uint i=0; i<contours.size(); ++i )
    {
        double currSize = cv::contourArea( contours[i] );
        if ( maxSize<currSize )
        {
            maxSize = currSize;
            maxIndex = i;
        }
    }

    if (contours.size()>0)
    if (type_simplification == CV_CHAIN_APPROX_NONE)
    {
        for ( size_t i=0; i<contours[maxIndex].size(); i=i+6 )
            output_contour.push_back(contours[maxIndex][i]);
    }
    else
    {
        output_contour = contours[maxIndex];
    }

    return output_contour;
}

// -----------------------------------------------------------------------------------
std::vector<std::vector<cv::Point>> ContoursFromMasks(const std::vector<cv::Mat> &masks,
                                        int type_simplification)
// -----------------------------------------------------------------------------------
{
    std::vector<std::vector<cv::Point>> contours;

    for (size_t t=0; t<masks.size(); ++t)
    {
        cv::Mat msk = masks[t].clone();
        std::vector<cv::Point> contour = ContourFromMask(msk, type_simplification);
        contours.push_back(contour);
    }

    return contours;
}

// -----------------------------------------------------------------------------------
cv::Mat ReadImage(const std::string &filename, const int opencv_read_type,
                  const int over_size)
// -----------------------------------------------------------------------------------
{
    cv::Mat image = cv::imread(filename, opencv_read_type);

    if (over_size>0)
    {
        cv::Rect subim_rect( cv::Point(over_size, over_size), image.size() );
        cv::Mat overim = cv::Mat::zeros(image.rows+over_size*2, image.cols+over_size*2, image.type());
        image.copyTo(overim(subim_rect));
        return overim;
    }
    else
    {
        return image;
    }
}

// -----------------------------------------------------------------------------------
cv::Mat CreateOverlay(const cv::Mat &image, const cv::Mat &mask)
// -----------------------------------------------------------------------------------
{
    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    cv::multiply(1-mask/255, channels[0], channels[0]);
    cv::multiply(1-mask/255, channels[2], channels[2]);

    cv::Mat out_col;
    cv::merge(channels, out_col);

    return out_col;
}

// -----------------------------------------------------------------------------------
cv::Mat CreateWhiteBG(const cv::Mat &image, const cv::Mat &mask)
// -----------------------------------------------------------------------------------
{
    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    cv::Mat canvas = cv::Mat::zeros(image.size(), image.type()) + cv::Scalar(255,255,255);
    image.copyTo(canvas, mask);

    return canvas;
}
