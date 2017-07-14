#include "DynamicProgramming.h"
#include "ClosedContour.h"
#include "EnergyTerms.h"
#include "Node.h"
#include "RotatedRect.h"
#include "ContourWarper.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

TEST(NodeEnergyTest, UseNodesWithEnergies)
{
    const int space_side=3;
    roam::Node node_dummy_1(cv::Point(10,10), roam::Node::Params(space_side));
    roam::Node node_dummy_2(cv::Point(20,10), roam::Node::Params(space_side));
    roam::Node node_dummy_3(cv::Point(20,20), roam::Node::Params(space_side));
    roam::Node node_dummy_4(cv::Point(10,20), roam::Node::Params(space_side));

    const int lab_siz = node_dummy_1.GetLabelSpaceSize();
    ASSERT_TRUE(lab_siz==space_side*space_side);

    // Draw a rectangle
    cv::Mat canvas = cv::Mat::zeros(30,30,CV_8UC1);
    cv::rectangle(canvas, cv::Rect(10,10,10,10), cv::Scalar(255), cv::FILLED);

    cv::Ptr<roam::UnaryTerm> unaryDummy = roam::UnaryTerm::create("GRAD");
    unaryDummy->Init(canvas);

    node_dummy_1.AddUnaryTerm(unaryDummy);
    node_dummy_2.AddUnaryTerm(unaryDummy);
    node_dummy_3.AddUnaryTerm(unaryDummy);
    node_dummy_4.AddUnaryTerm(unaryDummy);
}
