#include "DynamicProgramming.h"
#include "ClosedContour.h"
#include "EnergyTerms.h"
#include "Node.h"
#include "RotatedRect.h"
#include "ContourWarper.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

TEST(CCTest, ClosedContour_FollowGradients)
{
    const int space_side=5;

    roam::ClosedContour contour(roam::ClosedContour::Params(space_side*space_side));
    std::list<roam::Node>& contour_nodes = contour.contour_nodes;

    roam::Node node_dummy_1(cv::Point(10,10), roam::Node::Params(space_side));
    roam::Node node_dummy_2(cv::Point(20,10), roam::Node::Params(space_side));
    roam::Node node_dummy_3(cv::Point(20,20), roam::Node::Params(space_side));
    roam::Node node_dummy_4(cv::Point(10,20), roam::Node::Params(space_side));

    // Draw a rectangle
    cv::Mat canvas = cv::Mat::zeros(30,30,CV_8UC1);
    cv::rectangle(canvas, cv::Rect(12,12,10,10), cv::Scalar(255), cv::FILLED);
    cv::Ptr<roam::UnaryTerm> unaryDummy = roam::GradientUnary::createUnaryTerm(roam::GradientUnary::Params(roam::GradientUnary::SOBEL, 3, 1.0));
    unaryDummy->Init(canvas);
    cv::Ptr<roam::PairwiseTerm> binaryDummy = roam::NormPairwise::createPairwiseTerm(roam::NormPairwise::Params(roam::NormPairwise::L2, .01f));
    binaryDummy->Init(cv::Mat(), cv::Mat());

    node_dummy_1.AddUnaryTerm(unaryDummy);
    node_dummy_1.AddPairwiseTerm(binaryDummy);
    node_dummy_2.AddUnaryTerm(unaryDummy);
    node_dummy_2.AddPairwiseTerm(binaryDummy);
    node_dummy_3.AddUnaryTerm(unaryDummy);
    node_dummy_3.AddPairwiseTerm(binaryDummy);
    node_dummy_4.AddUnaryTerm(unaryDummy);
    node_dummy_4.AddPairwiseTerm(binaryDummy);

    contour_nodes.push_back(node_dummy_1);
    contour_nodes.push_back(node_dummy_2);
    contour_nodes.push_back(node_dummy_3);
    contour_nodes.push_back(node_dummy_4);

    contour.SetForegroundSideFlag(true);

    contour.BuildDPTable();
    FLOAT_TYPE min_cost = contour.RunDPInference();
    contour.ApplyMoves();

    std::vector<roam::label> curr_sol = contour.GetCurrentSolution();

    ASSERT_TRUE(std::abs(min_cost - 0.35207713f)<0.001);
}

TEST(CCTest, ClosedContour_FollowL2Norm)
{
    const int space_side=5;

    roam::ClosedContour contour(roam::ClosedContour::Params(space_side*space_side));
    std::list<roam::Node>& contour_nodes = contour.contour_nodes;

    roam::Node node_dummy_1(cv::Point(10,10), roam::Node::Params(space_side));
    roam::Node node_dummy_2(cv::Point(20,10), roam::Node::Params(space_side));
    roam::Node node_dummy_3(cv::Point(20,20), roam::Node::Params(space_side));
    roam::Node node_dummy_4(cv::Point(10,20), roam::Node::Params(space_side));

    // Draw a rectangle
    cv::Mat canvas = cv::Mat::zeros(30,30,CV_8UC1);
    cv::rectangle(canvas, cv::Rect(12,12,10,10), cv::Scalar(255), cv::FILLED);
    cv::Ptr<roam::UnaryTerm> unaryDummy = roam::GradientUnary::createUnaryTerm(roam::GradientUnary::Params(roam::GradientUnary::SOBEL, 3, 0.0));
    unaryDummy->Init(canvas);
    cv::Ptr<roam::PairwiseTerm> binaryDummy = roam::NormPairwise::createPairwiseTerm(roam::NormPairwise::Params(roam::NormPairwise::L2, 1.0));
    binaryDummy->Init(cv::Mat(), cv::Mat());

    node_dummy_1.AddUnaryTerm(unaryDummy);
    node_dummy_1.AddPairwiseTerm(binaryDummy);
    node_dummy_2.AddUnaryTerm(unaryDummy);
    node_dummy_2.AddPairwiseTerm(binaryDummy);
    node_dummy_3.AddUnaryTerm(unaryDummy);
    node_dummy_3.AddPairwiseTerm(binaryDummy);
    node_dummy_4.AddUnaryTerm(unaryDummy);
    node_dummy_4.AddPairwiseTerm(binaryDummy);

    contour_nodes.push_back(node_dummy_1);
    contour_nodes.push_back(node_dummy_2);
    contour_nodes.push_back(node_dummy_3);
    contour_nodes.push_back(node_dummy_4);

    contour.SetForegroundSideFlag(true);

    for (int iter=1; iter<=5; ++iter)
    {
        contour.BuildDPTable();
        contour.RunDPInference();
        contour.ApplyMoves();
    }

    for (auto itc=contour_nodes.begin(); itc!=--contour_nodes.end(); ++itc)
    {
        ASSERT_EQ(itc->GetCoordinates().x, itc->GetCoordinates().y);
        ASSERT_EQ(itc->GetCoordinates().x, std::next(itc,1)->GetCoordinates().x);
    }
}
