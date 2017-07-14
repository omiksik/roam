#include <iostream>
#include <exception>

#include "../tools/om_utils/include/roam/utils/timer.h"
#include "../roam/include/VideoSegmenter.h"
#include "../roam/include/Configuration.h"
#include "../roam/include/ClosedContour.h"

#include "main_utils.h"


// Command line options
const cv::String keys =
    "{help h usage ?       |                     | print this message             }"
    "{path_to_sequence seq |../Toy/Toy.txt       | path to sequence file          }"
    "{path_to_masks    msk |../Toy/ToyMasks.txt  | path to masks file             }"
    "{path_to_output   out |./output/Toy         | path to output folder          }"
    "{contour_width    wid |9                    | path to output folder          }"
    ;

int main(int argc, char *argv[])
{

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////// Parsing stuff //////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Draw contours v1.0.0");

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    cv::String path_to_sequence     = parser.get<cv::String>("path_to_sequence");
    cv::String path_to_masks        = parser.get<cv::String>("path_to_masks");
    cv::String path_to_output       = parser.get<cv::String>("path_to_output");
    int contour_width               = parser.get<int>("contour_width");

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    std::vector<std::string> list_of_files;
    std::vector<cv::Mat> images;
    if (!path_to_sequence.empty())
    {
        list_of_files = ListOfFilenames(path_to_sequence);

        for (size_t file_index = 0; file_index<list_of_files.size(); ++file_index)
            images.push_back(ReadImage(list_of_files[file_index], cv::IMREAD_COLOR, 0));

        if (list_of_files.empty())
        {
            LOG_ERROR("main(): File list is empty: Nothing to do.");
            return 0;
        }
    }
    else
    {
        LOG_ERROR("main(): No file list: Nothing to do.");
        return -1;
    }

    std::vector<std::string> list_of_masks;
    std::vector<cv::Mat> init_masks;
    if (!path_to_masks.empty())
    {
        list_of_masks = ListOfFilenames(path_to_masks);

        for (size_t file_index = 0; file_index<list_of_masks.size(); ++file_index)
            init_masks.push_back(ReadImage(list_of_masks[file_index], cv::IMREAD_GRAYSCALE, 0));

        if (init_masks.empty())
        {
            LOG_ERROR("main(): Something wrong with initial mask file: "<<path_to_masks);
            return -1;
        }
    }
    else
    {
        LOG_ERROR("main(): No initial mask: Nothing to do");
        return -1;
    }


    /////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////// Process ////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////
    LOG_INFO("main(): Starting core process...");

    std::vector<std::vector<cv::Point> > contours = ContoursFromMasks(init_masks);


    LOG_INFO("main(): Contours computed ");

    for (size_t file_index = 0; file_index<images.size(); ++file_index)
    {
        cv::Mat output = images[file_index].clone();

        std::string output_mask_file = path_to_output + std::string("/cont_") + std::to_string(file_index) + std::string(".png");
        std::string output_over_file = path_to_output + std::string("/over_") + std::to_string(file_index) + std::string(".jpg");

        cv::Mat over = images[file_index].clone();
        if (!init_masks[file_index].empty())
        {
            over /= 2;
            images[file_index].copyTo(over, init_masks[file_index]);
        }


        for (size_t it=0; it<contours[file_index].size(); ++it)
        {
            if (it==contours[file_index].size()-1)
            {
                cv::line(output, contours[file_index][it], contours[file_index][0], cv::Scalar(0,255,255),contour_width);
                cv::line(over, contours[file_index][it], contours[file_index][0], cv::Scalar(0,255,255),contour_width);
            }
            else
            {
                cv::line(output, contours[file_index][it], contours[file_index][it+1], cv::Scalar(0,255,255),contour_width, cv::LINE_AA);
                cv::line(over, contours[file_index][it], contours[file_index][it+1], cv::Scalar(0,255,255),contour_width, cv::LINE_AA);
            }
        }

        cv::imwrite(output_mask_file, output);
        cv::imwrite(output_over_file, over);
    }

    LOG_INFO("main(): DONE!");
    return 0;
}

