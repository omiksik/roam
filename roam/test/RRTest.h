#include "DynamicProgramming.h"
#include "ClosedContour.h"
#include "EnergyTerms.h"
#include "Node.h"
#include "RotatedRect.h"
#include "ContourWarper.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

static void drawRoamLine(cv::Mat& canvas, const roam::Line& line, cv::Scalar color)
{
    for (int x=0; x<canvas.cols; ++x)
        cv::circle(canvas, cv::Point2f(x,line.Y(x)),1,color);
}


TEST(RRTest, RotatedRectCont)
{
    cv::Mat canvas = cv::Mat::zeros(200,200,CV_8UC3);
    cv::rectangle(canvas, cv::Rect(10,10,180,180),cv::Scalar(128,128,128), -1);

    std::vector<cv::Point2f> cont = {cv::Point2f(80,20),cv::Point2f(100,20),cv::Point2f(120,50),
                                     cv::Point2f(100,80),cv::Point2f(80,80),cv::Point2f(70,60)};


    std::vector<roam::RotatedRect> roam_rects;
    roam_rects.reserve(cont.size());

    for (auto i=0; i<cont.size(); ++i)
    {
        const cv::Point2f &init = cont[i];
        const cv::Point2f &endit = cont[(i+1)%6];

        if (cont[i].x < cont[(i+1)%6].x)
        {
            roam_rects.push_back( roam::RotatedRect(init,endit,10,true) );
        }
        else
        {
            roam_rects.push_back( roam::RotatedRect(init,endit,10,false) );
        }

        std::vector<cv::Vec3f> colors;
        std::vector<FLOAT_TYPE> distances;
        std::vector<cv::Point> points;
        roam_rects[i].BuildDistanceAndColorVectors(canvas, colors, distances, points, true);
    }


}

TEST(RRTest, RotatedRectAll)
{
    cv::Mat canvas = cv::Mat::zeros(200,200,CV_8UC3);

    cv::Point2f p1(80, 100), p2(100, 100);

    roam::RotatedRect roam_rr1(p1, p2, 50, true);
    roam::RotatedRect roam_rr2(p1, p2, 50, false);


    cv::RotatedRect cv_rr1 = roam_rr1.GetCVRotatedRect();
    cv::RotatedRect cv_rr2 = roam_rr2.GetCVRotatedRect();

    cv::Point2f cv_rr_pts1[4], cv_rr_pts2[4];
    cv_rr1.points(cv_rr_pts1);
    cv_rr2.points(cv_rr_pts2);

    for (int i = 0; i < 4; i++)
        cv::line(canvas, cv_rr_pts1[i], cv_rr_pts1[(i+1)%4], cv::Scalar(0,255,0),3,-1);

    for (int i = 0; i < 4; i++)
        cv::line(canvas, cv_rr_pts2[i], cv_rr_pts2[(i+1)%4], cv::Scalar(0,0,255),3,-1);

    cv::Point2f rr1_pa, rr1_pb, rr1_pc, rr1_pd;
    cv::Point2f rr2_pa, rr2_pb, rr2_pc, rr2_pd;
    roam_rr1.GetCorners(rr1_pa, rr1_pb, rr1_pc, rr1_pd);
    roam_rr2.GetCorners(rr2_pa, rr2_pb, rr2_pc, rr2_pd);
    roam::Line lAB1, lBC1, lCD1, lDA1;
    roam::Line lAB2, lBC2, lCD2, lDA2;

    roam_rr1.GetLines(lAB1, lBC1, lCD1, lDA1);
    roam_rr2.GetLines(lAB2, lBC2, lCD2, lDA2);
}
