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
    "{@initial_mask    ini |../mask_Toy.png      | binary mask for first frame.   }"
    "{path_to_sequence seq |../Toy/Toy.txt       | path to sequence file          }"
    "{path_to_config   con |./default_exp.yaml   | path to config file            }"
    "{path_to_output   out |./output/Toy         | path to output folder          }"
    "{over_window_size win |0                    | outer window                   }"
    ;

int main(int argc, char *argv[])
{

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////// Parsing stuff //////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Fast roam CLI v1.0.0");

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    cv::String path_to_initial_mask = parser.get<cv::String>(0);
    cv::String path_to_config       = parser.get<cv::String>("path_to_config");
    cv::String path_to_sequence     = parser.get<cv::String>("path_to_sequence");
    cv::String path_to_output       = parser.get<cv::String>("path_to_output");
    int over_window_size            = parser.get<int>("over_window_size");

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    std::vector<std::string> list_of_files;
    if (!path_to_sequence.empty())
    {
        list_of_files = ListOfFilenames(path_to_sequence);

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

    ROAM::VideoSegmenter::Params video_segmenter_params;
    if (!path_to_config.empty())
    {
        cv::FileStorage fs(path_to_config, cv::FileStorage::READ);
        if (fs.isOpened())
        {
            try
            {
                video_segmenter_params.Read(fs);
            }
            catch (std::exception& e)
            {
                LOG_ERROR(e.what());
                return -1;
            }
        }
        else
            LOG_WARNING("main(): The file "<<path_to_config<<" could not be openend. Continuing with default params.");
    }

    cv::Mat init_mask;
    if (!path_to_initial_mask.empty())
    {
        init_mask = ReadImage(path_to_initial_mask, cv::IMREAD_GRAYSCALE, over_window_size);

        if (init_mask.empty())
        {
            LOG_ERROR("main(): Something wrong with initial mask file: "<<path_to_initial_mask);
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

    cv::Mat current_binary_mask = init_mask.clone();
    std::vector<cv::Point> current_contour = ContourFromMask(current_binary_mask);

    ROAM::VideoSegmenter video_segmenter(video_segmenter_params);

    if (!path_to_output.empty())
        video_segmenter.WriteOutput(path_to_output);

    const int num_iters = 1;
    for (size_t file_index = 0; file_index<list_of_files.size()-1; ++file_index)
    {
        LOG_INFO("*******************************************************");
        LOG_INFO("main(): Processing file: "<<list_of_files[file_index]);
        LOG_INFO("*******************************************************");

        cv::Mat prev_frame = ReadImage(list_of_files[ file_index ], cv::IMREAD_COLOR, over_window_size);
        cv::Mat next_frame = ReadImage(list_of_files[file_index+1], cv::IMREAD_COLOR, over_window_size);

        video_segmenter.SetPrevImage(prev_frame);
        video_segmenter.SetNextImage(next_frame);

        for (int i=0; i<num_iters; ++i)
        {
            video_segmenter.SetContours(current_contour);
            current_contour = video_segmenter.ProcessFrame();
        }

        video_segmenter.WritingOperations();
    }

    LOG_INFO("main(): DONE!");
    return 0;
}
