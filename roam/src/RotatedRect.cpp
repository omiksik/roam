#include "RotatedRect.h"

using namespace ROAM;

// -----------------------------------------------------------------------------------
Line::Line(const FLOAT_TYPE m, const FLOAT_TYPE b)
// -----------------------------------------------------------------------------------
{
    this->m = m;
    this->b = b;
}

// -----------------------------------------------------------------------------------
Line::Line(const cv::Point2f &p1, const cv::Point2f &p2)
// -----------------------------------------------------------------------------------
{
    this->m = FLOAT_TYPE(p1.y-p2.y) / FLOAT_TYPE(p1.x-p2.x);
    this->b = FLOAT_TYPE(p1.y) - this->m*p1.x;
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE Line::Y(const FLOAT_TYPE x) const
// -----------------------------------------------------------------------------------
{
    return this->m * x + this->b;
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE Line::M() const
// -----------------------------------------------------------------------------------
{
    return this->m; 
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE Line::B() const
// -----------------------------------------------------------------------------------
{
    return this->b;
}

// -----------------------------------------------------------------------------------
Line Line::perpLinePassPoint(const Line &perp_to, const cv::Point &p1)
// -----------------------------------------------------------------------------------
{
    const FLOAT_TYPE m = static_cast<FLOAT_TYPE>(-1.0 / (perp_to.M() + std::numeric_limits<FLOAT_TYPE>::epsilon()));
    const FLOAT_TYPE b = static_cast<FLOAT_TYPE>(p1.y - m * p1.x);
    return Line(m, b);
}

// -----------------------------------------------------------------------------------
#if defined(_MSC_VER) && (_MSC_VER <= 1800) // fix MSVC partial implementation of constexpr
    const FLOAT_TYPE RotatedRect::inner_rect_threshold = static_cast<FLOAT_TYPE>(0.01);
#endif
// -----------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
RotatedRect::RotatedRect()
// -----------------------------------------------------------------------------------
{
    this->hack_flash = false;
    this->pA.x = this->pB.x = this->pC.x = this->pD.x = 0;
    this->pA.y = this->pB.y = this->pC.y = this->pD.y = 0;
    this->rect_up = true;
    this->half_perimeter = 0.f;
}

// -----------------------------------------------------------------------------------
RotatedRect::RotatedRect(const cv::Rect& cv_rect)
// -----------------------------------------------------------------------------------
{
    this->hack_flash = false;
    this->pD = cv::Point2f(cv_rect.tl()) + cv::Point2f(0, static_cast<float>(cv_rect.height));
    this->pC = this->pD + cv::Point2f(static_cast<float>(cv_rect.width), 0);
    const FLOAT_TYPE height = static_cast<FLOAT_TYPE>(cv_rect.height);

    *this = RotatedRect(this->pD, this->pC, height, true);
}


// -----------------------------------------------------------------------------------
RotatedRect::RotatedRect(const cv::Point2f &po_1, const cv::Point2f &po_2,
                         const FLOAT_TYPE height, const bool rect_up)
// -----------------------------------------------------------------------------------
{

    this->hack_flash = false;
    this->rect_up = rect_up;

    this->pD = po_1;
    this->pC = po_2;

    //std::cerr<<pD.x<<"=="<<pC.x<<" AND "<<pD.y<<"=="<<pC.y<<std::endl;

    if(std::abs(pD.x-pC.x) < 0.0001)
    {
        pD.x += 0.0001f;
        hack_flash = true;
    }
    if(std::abs(pD.y - pC.y) < 0.0001)
    {
        pD.y += 0.0001f;
        hack_flash = true;
    }

    this->lCD = Line(this->pC, this->pD);
    this->lBC = Line::perpLinePassPoint(this->lCD, this->pC);
    this->lDA = Line::perpLinePassPoint(this->lCD, this->pD);

    if (this->rect_up)
        this->lAB = Line(this->lCD.M(), lCD.B()-height*std::sqrt(lCD.M()*lCD.M()+1));
    else
        this->lAB = Line(this->lCD.M(), lCD.B()+height*std::sqrt(lCD.M()*lCD.M()+1));

    const FLOAT_TYPE xA = (lDA.B()-lAB.B()) / (lAB.M() - lDA.M());
    const FLOAT_TYPE xB = (lBC.B()-lAB.B()) / (lAB.M() - lBC.M());
    this->pA = cv::Point2f(xA,lAB.Y(xA));
    this->pB = cv::Point2f(xB,lAB.Y(xB));

    this->half_perimeter = static_cast<FLOAT_TYPE>( cv::norm(this->pA-this->pB) + cv::norm(this->pB-this->pC) );
}

// -----------------------------------------------------------------------------------
cv::Vec3d RotatedRect::SumOver(const LineIntegralImage &verticalLineIntegralImage_) const
// -----------------------------------------------------------------------------------
{
    const cv::Mat& verticalLineIntegralImage = verticalLineIntegralImage_.data;

    std::vector<cv::Point2f> ups, dos;

    if(this->pA.x <= this->pB.x && this->pA.x <= this->pC.x && this->pA.x <= this->pD.x)
    {
        // A is leftmost
        for(int x = static_cast<int>(pA.x); x < static_cast<int>(pD.x); ++x)
        {
            ups.push_back(cv::Point2f(static_cast<float>(x), lAB.Y(static_cast<float>(x))));
            dos.push_back(cv::Point2f(static_cast<float>(x), lDA.Y(static_cast<float>(x))));
        }

        for (int x = static_cast<int>(pD.x); x < static_cast<int>(pB.x); ++x)
        {
            ups.push_back(cv::Point2f(static_cast<float>(x), lAB.Y(static_cast<float>(x))));
            dos.push_back(cv::Point2f(static_cast<float>(x), lCD.Y(static_cast<float>(x))));
        }

        for (int x = static_cast<int>(pB.x); x < static_cast<int>(pC.x); ++x)
        {
            ups.push_back(cv::Point2f(static_cast<float>(x), lBC.Y(static_cast<float>(x))));
            dos.push_back(cv::Point2f(static_cast<float>(x), lCD.Y(static_cast<float>(x))));
        }
    }

    if(this->pB.x<this->pA.x && this->pB.x<=this->pC.x && this->pB.x<=this->pD.x  )
    {
        // B is leftmost
        for(int x = static_cast<int>(pB.x); x<static_cast<int>(pC.x); ++x)
        {
            ups.push_back(cv::Point2f(static_cast<float>(x), lBC.Y(static_cast<float>(x))));
            dos.push_back(cv::Point2f(static_cast<float>(x), lAB.Y(static_cast<float>(x))));
        }

        for(int x = static_cast<int>(pC.x); x<static_cast<int>(pA.x); ++x)
        {
            ups.push_back(cv::Point2f(static_cast<float>(x), lCD.Y(static_cast<float>(x))));
            dos.push_back(cv::Point2f(static_cast<float>(x), lAB.Y(static_cast<float>(x))));
        }

        for(int x = static_cast<int>(pA.x); x<static_cast<int>(pD.x); ++x)
        {
            ups.push_back(cv::Point2f(static_cast<float>(x), lCD.Y(static_cast<float>(x))));
            dos.push_back(cv::Point2f(static_cast<float>(x), lDA.Y(static_cast<float>(x))));
        }
    }

    if ( this->pC.x<this->pB.x && this->pC.x<this->pA.x && this->pC.x<=this->pD.x  )
    {
        // C is leftmost
        for(int x = static_cast<int>(pC.x); x<static_cast<int>(pB.x); ++x)
        {
            ups.push_back(cv::Point2f(static_cast<float>(x), lCD.Y(static_cast<float>(x))));
            dos.push_back(cv::Point2f(static_cast<float>(x), lBC.Y(static_cast<float>(x))));
        }

        for(int x = static_cast<int>(pB.x); x<static_cast<int>(pD.x); ++x)
        {
            ups.push_back(cv::Point2f(static_cast<float>(x), lCD.Y(static_cast<float>(x))));
            dos.push_back(cv::Point2f(static_cast<float>(x), lAB.Y(static_cast<float>(x))));
        }

        for(int x = static_cast<int>(pD.x); x<static_cast<int>(pA.x); ++x)
        {
            ups.push_back(cv::Point2f(static_cast<float>(x), lDA.Y(static_cast<float>(x))));
            dos.push_back(cv::Point2f(static_cast<float>(x), lAB.Y(static_cast<float>(x))));
        }
    }

    if ( this->pD.x<this->pB.x && this->pD.x<this->pC.x && this->pD.x<this->pA.x  )
    {
        // D is leftmost
        for(int x = static_cast<int>(pD.x); x < static_cast<int>(pA.x); ++x)
        {
            ups.push_back(cv::Point2f(static_cast<float>(x), lDA.Y(static_cast<float>(x))));
            dos.push_back(cv::Point2f(static_cast<float>(x), lCD.Y(static_cast<float>(x))));
        }

        for(int x = static_cast<int>(pA.x); x < static_cast<int>(pC.x); ++x)
        {
            ups.push_back(cv::Point2f(static_cast<float>(x), lAB.Y(static_cast<float>(x))));
            dos.push_back(cv::Point2f(static_cast<float>(x), lCD.Y(static_cast<float>(x))));
        }

        for(int x = static_cast<int>(pC.x); x < static_cast<int>(pB.x); ++x)
        {
            ups.push_back(cv::Point2f(static_cast<float>(x), lAB.Y(static_cast<float>(x))));
            dos.push_back(cv::Point2f(static_cast<float>(x), lBC.Y(static_cast<float>(x))));
        }
    }

    // Do the Sum
    if (verticalLineIntegralImage.type() == CV_32SC3)
    {
        cv::Vec3d output(0, 0, 0);

        // TODO: can be parallelized
        for(auto c=0; c<dos.size(); ++c)
        {
            if(std::abs(dos[c].x-ups[c].x) > half_perimeter || std::abs(dos[c].y-ups[c].y) > half_perimeter)
                continue;

            if(dos[c].x<0 || dos[c].y<0 || dos[c].x >= verticalLineIntegralImage.cols || dos[c].y >= verticalLineIntegralImage.rows)
                continue;

            if(ups[c].x<0 || ups[c].y<0 || ups[c].x >= verticalLineIntegralImage.cols || ups[c].y >= verticalLineIntegralImage.rows)
                continue;

            output += ( verticalLineIntegralImage.at<cv::Vec3d>(dos[c]) - verticalLineIntegralImage.at<cv::Vec3d>(ups[c]) );
        }

        return output;
    }
    else
    if (verticalLineIntegralImage.type()==CV_32SC1)
    {
        double output = 0.0;

        #pragma omp parallel for reduction(+:output)
        for(auto c = 0; c<dos.size(); ++c)
        {
            if(std::abs(dos[c].x-ups[c].x) > half_perimeter || std::abs(dos[c].y-ups[c].y) > half_perimeter)
                continue;

            if(dos[c].x < 0 || dos[c].y < 0 || dos[c].x>=verticalLineIntegralImage.cols || dos[c].y >= verticalLineIntegralImage.rows)
                continue;

            if(ups[c].x < 0 || ups[c].y < 0 || ups[c].x>=verticalLineIntegralImage.cols || ups[c].y >= verticalLineIntegralImage.rows)
                continue;

            output += ( verticalLineIntegralImage.at<int>(dos[c]) - verticalLineIntegralImage.at<int>(ups[c]) );
        }
        return cv::Vec3d(output, output, output);
    }
    else
    if (verticalLineIntegralImage.type()==CV_64FC1)
    {
        double output = 0.0;
        
        #pragma omp parallel for reduction(+:output)
        for(auto c = 0; c < dos.size(); ++c)
        {
            if (std::abs(dos[c].x - ups[c].x) > half_perimeter || std::abs(dos[c].y - ups[c].y) > half_perimeter)
                continue;

            if (dos[c].x < 0 || dos[c].y < 0 || dos[c].x >= verticalLineIntegralImage.cols || dos[c].y >= verticalLineIntegralImage.rows)
                continue;

            if (ups[c].x < 0 || ups[c].y < 0 || ups[c].x >= verticalLineIntegralImage.cols || ups[c].y >= verticalLineIntegralImage.rows)
                continue;

            output += ( verticalLineIntegralImage.at<double>(dos[c]) - verticalLineIntegralImage.at<double>(ups[c]) );
        }

        return cv::Vec3d(output, output, output);
    }

    return cv::Vec3d(0.0, 0.0, 0.0);
}

// -----------------------------------------------------------------------------------
cv::RotatedRect RotatedRect::GetCVRotatedRect() const
// -----------------------------------------------------------------------------------
{
    const cv::Point2f center = (pA + pB + pC + pD) / 4.f;
    const FLOAT_TYPE angle = static_cast<FLOAT_TYPE>(std::atan(lCD.M()) * 180.f / CV_PI);
    const cv::Size2f size(static_cast<float>(cv::norm(pC - pD)), static_cast<float>(cv::norm(pA - pD)));
    return cv::RotatedRect(center, size, angle);
}

// -----------------------------------------------------------------------------------
void RotatedRect::BuildDistanceAndColorVectors(const cv::Mat &reference_image,
        std::vector<cv::Vec3f> &colors, std::vector<FLOAT_TYPE> &distances_line_cd, std::vector<cv::Point> &points,
        bool build_points) const
// -----------------------------------------------------------------------------------
{
    const cv::Rect whole_im(0, 0, reference_image.cols, reference_image.rows);
    const cv::Rect bound_re = this->GetCVRotatedRect().boundingRect();

    const cv::Rect valid_re = whole_im & bound_re;

    colors.clear();
    distances_line_cd.clear();
    colors.reserve(static_cast<size_t>(valid_re.area()));
    distances_line_cd.reserve(static_cast<size_t>(valid_re.area()));

    if (build_points)
    {
        points.clear();
        points.reserve(static_cast<size_t>(valid_re.area()));
    }

    for (int y = valid_re.y; y < valid_re.y + valid_re.height; ++y)
    {
        const cv::Vec3b* ptr_row = reference_image.ptr<cv::Vec3b>(y);

        for (int x = valid_re.x; x < valid_re.x+valid_re.width; ++x)
        {
            const FLOAT_TYPE d_cd = std::abs(-lCD.M() * x + y - lCD.B()) / std::sqrt(1+lCD.M() * lCD.M());

            if (!hack_flash)
            {
                const FLOAT_TYPE d_ab = std::abs(-lAB.M() * x + y - lAB.B()) / std::sqrt(1 + lAB.M() * lAB.M());
                const FLOAT_TYPE d_bc = std::abs(-lBC.M() * x + y - lBC.B()) / std::sqrt(1 + lBC.M() * lBC.M());
                const FLOAT_TYPE d_da = std::abs(-lDA.M() * x + y - lDA.B()) / std::sqrt(1 + lDA.M() * lDA.M());

                if (std::abs((d_ab + d_bc + d_cd + d_da) - half_perimeter ) <= inner_rect_threshold)
                {
                    colors.push_back(cv::Vec3f(ptr_row[x][0], ptr_row[x][1], ptr_row[x][2]));
                    distances_line_cd.push_back(d_cd);

                    if (build_points)
                        points.push_back(cv::Point(x,y));
                }
            }
            else
            {
                colors.push_back(cv::Vec3f(ptr_row[x][0], ptr_row[x][1], ptr_row[x][2]));
                distances_line_cd.push_back(d_cd);

                if (build_points)
                    points.push_back( cv::Point(x,y) );
            }
        }
    }
}

// -----------------------------------------------------------------------------------
void RotatedRect::BuildDistanceAndColorVectors(const cv::Mat &reference_image, const cv::Mat& valid_mask, const unsigned char mask_label,
        std::vector<cv::Vec3f> &colors, std::vector<FLOAT_TYPE> &distances_line_cd,
                                               std::vector<cv::Point> &points, bool build_points) const
// -----------------------------------------------------------------------------------
{
    assert(valid_mask.size()==reference_image.size());
    assert(valid_mask.type() == CV_8UC1);

    const cv::Rect whole_im(0, 0, reference_image.cols, reference_image.rows);
    const cv::Rect bound_re = this->GetCVRotatedRect().boundingRect();

    const cv::Rect valid_re = whole_im & bound_re;

    colors.clear();
    distances_line_cd.clear();
    colors.reserve(static_cast<size_t>(valid_re.area()));
    distances_line_cd.reserve(static_cast<size_t>(valid_re.area()));

    if (build_points)
    {
        points.clear();
        points.reserve(static_cast<size_t>(valid_re.area()));
    }

    for (int y = valid_re.y; y < valid_re.y + valid_re.height; ++y)
    {
        const cv::Vec3b* ptr_row = reference_image.ptr<cv::Vec3b>(y);
        const unsigned char* msk_row = valid_mask.ptr<unsigned char>(y);

        for (int x = valid_re.x; x < valid_re.x + valid_re.width; ++x)
        {
            if (msk_row[x]!=mask_label) continue;

            const FLOAT_TYPE d_cd = std::abs(-lCD.M() * x + y - lCD.B()) / std::sqrt(1 + lCD.M() * lCD.M());

            if (!hack_flash)
            {
                const FLOAT_TYPE d_ab = std::abs(-lAB.M() * x + y - lAB.B()) / std::sqrt(1 + lAB.M() * lAB.M());
                const FLOAT_TYPE d_bc = std::abs(-lBC.M() * x + y - lBC.B()) / std::sqrt(1 + lBC.M() * lBC.M());
                const FLOAT_TYPE d_da = std::abs(-lDA.M() * x + y - lDA.B()) / std::sqrt(1 + lDA.M() * lDA.M());

                if ( std::abs( (d_ab+d_bc+d_cd+d_da) - half_perimeter ) <= inner_rect_threshold )
                {
                    colors.push_back( cv::Vec3f(ptr_row[x][0],ptr_row[x][1],ptr_row[x][2]) );
                    distances_line_cd.push_back( d_cd );

                    if (build_points)
                        points.push_back( cv::Point(x,y) );
                }
            }
            else
            {
                colors.push_back( cv::Vec3f(ptr_row[x][0],ptr_row[x][1],ptr_row[x][2]) );
                distances_line_cd.push_back( d_cd );
                if (build_points)
                    points.push_back( cv::Point(x,y) );
            }
        }
    }
}

// -----------------------------------------------------------------------------------
void RotatedRect::BuildDistanceAndColorVectors_2(const cv::Mat& reference_image, const cv::Mat& valid_mask, const unsigned char mask_label,
                                                 std::vector<cv::Vec3f> &colors, std::vector<FLOAT_TYPE>& distances_line_cd,
                                                 std::vector<cv::Point> &points,
                                                 bool build_points) const
// -----------------------------------------------------------------------------------
{
    const cv::Rect whole_im(0, 0, reference_image.cols, reference_image.rows);
    const cv::Rect bound_re = this->GetCVRotatedRect().boundingRect();

    const cv::Rect valid_re = whole_im & bound_re;

    colors.clear();
    distances_line_cd.clear();
    colors.reserve(static_cast<size_t>(valid_re.area()));
    distances_line_cd.reserve(static_cast<size_t>(valid_re.area()));

    if (build_points)
    {
        points.clear();
        points.reserve(static_cast<size_t>(valid_re.area()));
    }

    Line l1, l2, l3, l4;
    const FLOAT_TYPE angle_lab = std::atan(lAB.M());
    FLOAT_TYPE d_l1_l2;
    if (angle_lab > CV_PI / 4)
    {
        l1 = lDA;
        l2 = lBC;
        l3 = lAB;
        l4 = lCD;
        d_l1_l2 = static_cast<FLOAT_TYPE>(cv::norm(pA - pB));
    }
    else
    {
        l1 = lAB;
        l2 = lCD;
        l3 = lDA;
        l4 = lBC;
        d_l1_l2 = static_cast<FLOAT_TYPE>(cv::norm(pA - pD));
    }

    for (FLOAT_TYPE d_inc = 0; d_inc < d_l1_l2; d_inc += 1.f)
    {
        Line curr_l;
        if (d_inc==0)
            curr_l = l1;
        else
        {
            const Line lt1(l1.M(), l1.B() - d_inc*std::sqrt(l1.M()*l1.M() + 1));
            const Line lt2(l1.M(), l1.B() + d_inc*std::sqrt(l1.M()*l1.M() + 1));

            const FLOAT_TYPE d_lt1_l1 = std::abs(-lt1.B() + l1.B());
            const FLOAT_TYPE d_lt1_l2 = std::abs(-lt1.B() + l2.B());

            const FLOAT_TYPE d_lt2_l1 = std::abs(-lt2.B() + l1.B());
            const FLOAT_TYPE d_lt2_l2 = std::abs(-lt2.B() + l2.B());

            if (d_lt1_l1+d_lt1_l2 < d_lt2_l1+d_lt2_l2)
                curr_l = lt1;
            else
                curr_l = lt2;
        }

        const FLOAT_TYPE xi = std::min((l3.B() - curr_l.B()) / (curr_l.M() - l3.M()), (l4.B() - curr_l.B()) / (curr_l.M() - l4.M()));
        const FLOAT_TYPE xe = std::max((l3.B() - curr_l.B()) / (curr_l.M() - l3.M()), (l4.B() - curr_l.B()) / (curr_l.M() - l4.M()));

        for (FLOAT_TYPE x=xi; x<xe; ++x)
        {
            const FLOAT_TYPE y=curr_l.Y(x);
            if (x<0 || x>reference_image.cols || y<0 || y>reference_image.rows)
                continue;

            cv::Point p(static_cast<int>(x), static_cast<int>(y));

            if (valid_mask.at<unsigned char>(p)!=mask_label) 
                continue;

            if (build_points)
                points.push_back(p);

            const cv::Vec3b colo = reference_image.at<cv::Vec3b>(p);
            colors.push_back(cv::Vec3f(colo[0], colo[1], colo[2]));

            const FLOAT_TYPE d_cd = std::abs(-lCD.M()*x + y - lCD.B()) / std::sqrt(1 + lCD.M()*lCD.M());
            distances_line_cd.push_back(d_cd);
        }
    }
}

// -----------------------------------------------------------------------------------
void RotatedRect::BuildDistanceAndColorVectors_2(const cv::Mat &reference_image,
        std::vector<cv::Vec3f> &colors, std::vector<FLOAT_TYPE> &distances_line_cd, std::vector<cv::Point> &points,
        bool build_points) const
// -----------------------------------------------------------------------------------
{
    const cv::Rect whole_im(0, 0, reference_image.cols, reference_image.rows);
    const cv::Rect bound_re = this->GetCVRotatedRect().boundingRect();

    const cv::Rect valid_re = whole_im & bound_re;

    colors.clear();
    distances_line_cd.clear();
    colors.reserve(static_cast<size_t>(valid_re.area()));
    distances_line_cd.reserve(static_cast<size_t>(valid_re.area()));

    if (build_points)
    {
        points.clear();
        points.reserve(static_cast<size_t>(valid_re.area()));
    }

    Line l1, l2, l3, l4;
    const FLOAT_TYPE angle_lab = std::atan(lAB.M());
    FLOAT_TYPE d_l1_l2;
    if (angle_lab > CV_PI/4)
    {
        l1 = lDA;
        l2 = lBC;
        l3 = lAB;
        l4 = lCD;
        d_l1_l2 = static_cast<FLOAT_TYPE>(cv::norm(pA - pB));
    }
    else
    {
        l1 = lAB;
        l2 = lCD;
        l3 = lDA;
        l4 = lBC;
        d_l1_l2 = static_cast<FLOAT_TYPE>(cv::norm(pA - pD));
    }

    for (FLOAT_TYPE d_inc=0; d_inc<d_l1_l2; d_inc+=1.f)
    {
        Line curr_l;
        if (d_inc==0)
            curr_l = l1;
        else
        {
            const Line lt1(l1.M(), l1.B() - d_inc*std::sqrt(l1.M()*l1.M() + 1));
            const Line lt2(l1.M(), l1.B() + d_inc*std::sqrt(l1.M()*l1.M() + 1));

            const FLOAT_TYPE d_lt1_l1 = std::abs(-lt1.B() + l1.B());
            const FLOAT_TYPE d_lt1_l2 = std::abs(-lt1.B() + l2.B());

            const FLOAT_TYPE d_lt2_l1 = std::abs(-lt2.B() + l1.B());
            const FLOAT_TYPE d_lt2_l2 = std::abs(-lt2.B() + l2.B());

            if (d_lt1_l1+d_lt1_l2 < d_lt2_l1+d_lt2_l2)
                curr_l = lt1;
            else
                curr_l = lt2;
        }

        const FLOAT_TYPE xi = std::min((l3.B() - curr_l.B()) / (curr_l.M() - l3.M()), (l4.B() - curr_l.B()) / (curr_l.M() - l4.M()));
        const FLOAT_TYPE xe = std::max((l3.B() - curr_l.B()) / (curr_l.M() - l3.M()), (l4.B() - curr_l.B()) / (curr_l.M() - l4.M()));

        for(FLOAT_TYPE x = xi; x < xe; ++x)
        {
            const FLOAT_TYPE y = curr_l.Y(x);
            if(x < 0 || x > reference_image.cols || y < 0 || y > reference_image.rows)
                continue;

            cv::Point p(static_cast<int>(x), static_cast<int>(y));
            if(build_points)
                points.push_back(p);

            const cv::Vec3b colo = reference_image.at<cv::Vec3b>(p);
            colors.push_back(cv::Vec3f(colo[0],colo[1],colo[2]));

            const FLOAT_TYPE d_cd = std::abs(-lCD.M()*x + y - lCD.B()) / std::sqrt(1 + lCD.M()*lCD.M());
            distances_line_cd.push_back(d_cd);
        }
    }
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE RotatedRect::Area() const
// -----------------------------------------------------------------------------------
{
    return static_cast<FLOAT_TYPE>(cv::norm(this->pA - this->pB) * cv::norm(this->pB - this->pC));
}

// -----------------------------------------------------------------------------------
void RotatedRect::GetCorners(cv::Point2f &pA, cv::Point2f &pB,
                             cv::Point2f &pC, cv::Point2f &pD) const
// -----------------------------------------------------------------------------------
{
    pA = this->pA;
    pB = this->pB;
    pC = this->pC;
    pD = this->pD;
}

// -----------------------------------------------------------------------------------
void RotatedRect::GetLines(Line &_lAB, Line &_lBC, Line &_lCD, Line &_lDA) const
// -----------------------------------------------------------------------------------
{
    _lAB = this->lAB;
    _lBC = this->lBC;
    _lCD = this->lCD;
    _lDA = this->lDA;
}

// -----------------------------------------------------------------------------------
LineIntegralImage LineIntegralImage::CreateFromImage(const cv::Mat &image)
// -----------------------------------------------------------------------------------
{
    LineIntegralImage out;
    assert(image.type()==CV_8UC3 || image.depth()==CV_8UC1);

    if (image.type()==CV_8UC3)
    {
        out.data.create(image.size(), CV_32SC3);

        const cv::Vec3b* pixel_i = image.ptr<cv::Vec3b>(0);
        cv::Vec3i* data_p = out.data.ptr<cv::Vec3i>(0);

        #pragma omp parallel for
        for(auto c = 0; c < image.cols; ++c)
            data_p[c] = cv::Vec3i(pixel_i[c]);

        for (int r=1; r<image.rows; ++r)
        {
            const cv::Vec3i* prev_data =  out.data.ptr<cv::Vec3i>(r-1);

            pixel_i = image.ptr<cv::Vec3b>(r);
            data_p =  out.data.ptr<cv::Vec3i>(r);

            #pragma omp parallel for
            for(auto c = 0; c < image.cols; ++c)
                data_p[c] = prev_data[c] + cv::Vec3i(pixel_i[c]);
        }
    }
    else if (image.type()==CV_8UC1)
    {


        out.data.create(image.size(), CV_32SC1);

        const unsigned char* pixel_i = image.ptr<unsigned char>(0);
        int* data_p =  out.data.ptr<int>(0);

        #pragma omp parallel for
        for(auto c=0; c < image.cols; ++c)
            data_p[c] = static_cast<int>(pixel_i[c]);

        for (int r=1; r<image.rows; ++r)
        {
            const int* prev_data =  out.data.ptr<int>(r-1);

            pixel_i = image.ptr<unsigned char>(r);
            data_p =  out.data.ptr<int>(r);

            #pragma omp parallel for
            for(auto c=0; c<image.cols; ++c)
                data_p[c] = prev_data[c] + static_cast<int>(pixel_i[c]);
        }

    }
    else if (image.type()==CV_32FC1)
    {
        out.data.create(image.size(), CV_64FC1);

        const float* pixel_i = image.ptr<float>(0);
        double* data_p =  out.data.ptr<double>(0);

        #pragma omp parallel for
        for (auto c=0; c<image.cols; ++c)
            data_p[c] = static_cast<double>(pixel_i[c]);

        for (int r=1; r<image.rows; ++r)
        {
            const double* prev_data =  out.data.ptr<double>(r-1);

            pixel_i = image.ptr<float>(r);
            data_p =  out.data.ptr<double>(r);

            #pragma omp parallel for
            for (auto c=0; c<image.cols; ++c)
                data_p[c] = prev_data[c] + static_cast<double>(pixel_i[c]);
        }
    }

    return out;
}