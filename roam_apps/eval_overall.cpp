
#include <iostream>
#include <exception>

#include <opencv2/core.hpp>

#include "../tools/om_utils/include/roam/utils/timer.h"
#include "../tools/om_utils/include/roam/utils/confusion_matrix.h"
#include "../roam/include/VideoSegmenter.h"
#include "../roam/include/Configuration.h"
#include "../roam/include/ClosedContour.h"

#include "main_utils.h"

#define USE_CUDA

// Command line options
// # ./eval_cli -gt_files="./gt/gt.txt" -res_files="./res/res.txt" -out="./out.yaml"
const cv::String keys =
    "{help h usage ?       |                     | print this message             }"
    "{gt_files gt          |./gt.txt             | list of files gt               }"
    "{res_files res        |./res.txt            | list of result files           }"
    "{path_to_output   out |./out.yaml           | output file                    }"
    ;

int main(int argc, char *argv[])
{

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////// Parsing stuff //////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Evaluation CLI v1.0.0");

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    cv::String gt_filelist_name = parser.get<cv::String>("gt_files");
    cv::String res_filelist_name = parser.get<cv::String>("res_files");
    cv::String output_filename = parser.get<cv::String>("path_to_output");

    std::vector<std::string> list_of_gtfiles;
    if (!gt_filelist_name.empty())
        list_of_gtfiles = ListOfFilenames(gt_filelist_name);
    else
    {
        LOG_ERROR("main(): No filelist: Nothing to do");
        return 0;
    }

    std::vector<std::string> list_of_resfiles;
    if (!res_filelist_name.empty())
        list_of_resfiles = ListOfFilenames(res_filelist_name);
    else
    {
        LOG_ERROR("main(): No filelist: Nothing to do");
        return 0;
    }

    if (list_of_resfiles.size()!=list_of_gtfiles.size())
    {
        LOG_ERROR("main(): Number of gt files and result files differ: "<<list_of_resfiles.size()<<"!="<<list_of_gtfiles.size() <<": Aborting");
        return 0;
    }

    if (output_filename.empty())
    {
        LOG_ERROR("main(): No output file: Nothing to do");
        return 0;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////// Processing ///////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////
    ROAM::ConfusionMatrix cm_operator(2);

    //#pragma omp parallel for
    for(auto file_index = 0; file_index < list_of_gtfiles.size() - 1; ++file_index)
    {

        cv::Mat gt_frame = cv::imread(list_of_gtfiles[file_index],  cv::IMREAD_GRAYSCALE)/255;
        cv::Mat re_frame = cv::imread(list_of_resfiles[file_index], cv::IMREAD_GRAYSCALE)/255;

        std::vector<int> gt_values;
        std::vector<int> re_values;
        MatToVector<int>(gt_frame, gt_values);
        MatToVector<int>(re_frame, re_values);

        cm_operator.Accumulate(gt_values, re_values);
    }
    double iou = cm_operator.IoU(1);
    double acc = cm_operator.Accuracy();
    double f_1 = cm_operator.F1(1);
    double pre = cm_operator.Precision(1);
    double rec = cm_operator.Recall(1);

    cv::FileStorage output_fs(output_filename, cv::FileStorage::WRITE);
    output_fs << "mean_intersection_over_union" << iou;
    output_fs << "mean_accuracy" << acc;
    output_fs << "mean_f1score" << f_1;
    output_fs << "mean_precision" << pre;
    output_fs << "mean_recall" << rec;

    return 0;
}
