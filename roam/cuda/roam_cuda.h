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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "../include/Configuration.h"
#include "omp.h"
#include "cuda_profiler_api.h"

#ifdef __CUDACC__
#define CUDA_METHOD __host__ __device__
#else
#define CUDA_METHOD
#endif

//#define CUDA_DEBUG

#include "math.h"

namespace cuda_roam
{

// -----------------------------------------------------------------------------------
class Point
// -----------------------------------------------------------------------------------
{
public:
    // -----------------------------------------------------------------------------------
    CUDA_METHOD Point()
    // -----------------------------------------------------------------------------------
    {
        this->x=0;
        this->y=0;
    }

    // -----------------------------------------------------------------------------------
    CUDA_METHOD Point(const FLOAT_TYPE x, const FLOAT_TYPE y)
    // -----------------------------------------------------------------------------------
    {
        this->x=x;
        this->y=y;
    }

    // -----------------------------------------------------------------------------------
    CUDA_METHOD ~Point()
    // -----------------------------------------------------------------------------------
    {}

    FLOAT_TYPE x;
    FLOAT_TYPE y;

    // -----------------------------------------------------------------------------------
    CUDA_METHOD Point& operator+=(const Point& rhs)
    // -----------------------------------------------------------------------------------
    {
        this->x+=rhs.x;
        this->y+=rhs.y;
        return *this;
    }

    // -----------------------------------------------------------------------------------
    friend CUDA_METHOD Point operator+ (Point p1, const Point& p2)
    // -----------------------------------------------------------------------------------
    {
        p1 += p2;
        return p1;
    }

    // -----------------------------------------------------------------------------------
    CUDA_METHOD Point& operator-=(const Point& rhs)
    // -----------------------------------------------------------------------------------
    {
        this->x-=rhs.x;
        this->y-=rhs.y;
        return *this;
    }

    // -----------------------------------------------------------------------------------
    friend CUDA_METHOD Point operator- (Point p1, const Point& p2)
    // -----------------------------------------------------------------------------------
    {
        p1 -= p2;
        return p1;
    }
};


// -----------------------------------------------------------------------------------
class Line
// -----------------------------------------------------------------------------------
{
public:

    // -----------------------------------------------------------------------------------
    CUDA_METHOD explicit Line(const Point &p1, const Point &p2)
    // -----------------------------------------------------------------------------------
    {
        this->m = static_cast<FLOAT_TYPE>((p1.y-p2.y) / (p1.x-p2.x+1e-15));
        this->b = static_cast<FLOAT_TYPE>(p1.y - this->m*p1.x);
    }

    // -----------------------------------------------------------------------------------
    CUDA_METHOD explicit Line(const FLOAT_TYPE m = 1, const FLOAT_TYPE b = 0)
    // -----------------------------------------------------------------------------------
    {
        this->m = m;
        this->b = b;
    }

    // -----------------------------------------------------------------------------------
    CUDA_METHOD ~Line()
    // -----------------------------------------------------------------------------------
    {}

    // -----------------------------------------------------------------------------------
    CUDA_METHOD FLOAT_TYPE Y(const FLOAT_TYPE x) const
    // -----------------------------------------------------------------------------------
    {
        return this->m * x + this->b;
    }

    // -----------------------------------------------------------------------------------
    CUDA_METHOD FLOAT_TYPE M() const
    // -----------------------------------------------------------------------------------
    {
        return this->m;
    }

    // -----------------------------------------------------------------------------------
    CUDA_METHOD FLOAT_TYPE B() const
    // -----------------------------------------------------------------------------------
    {
        return this->b;
    }

    // -----------------------------------------------------------------------------------
    CUDA_METHOD static Line perpLinePassPoint(const Line& perp_to, const Point p1)
    // -----------------------------------------------------------------------------------
    {
        const FLOAT_TYPE m = static_cast<FLOAT_TYPE>(-1.0 / perp_to.M());
        const FLOAT_TYPE b = static_cast<FLOAT_TYPE>(p1.y - m * p1.x);
        return Line(m,b);
    }

protected:
    FLOAT_TYPE m;
    FLOAT_TYPE b;
};


static CUDA_METHOD FLOAT_TYPE norm(const Point p1)
{
    return sqrtf(p1.x*p1.x + p1.y*p1.y);
}

// -----------------------------------------------------------------------------------
class RotatedRect
// -----------------------------------------------------------------------------------
{
public:

    // -----------------------------------------------------------------------------------
    CUDA_METHOD RotatedRect()
    // -----------------------------------------------------------------------------------
    {
        this->hack_flash = false;
        this->pA.x = this->pB.x = this->pC.x = this->pD.x = 0;
        this->pA.y = this->pB.y = this->pC.y = this->pD.y = 0;
        this->rect_up = true;
        this->half_perimeter = 0.f;
        this->inner_rect_threshold = static_cast<FLOAT_TYPE>(0.01);
    }

    // -----------------------------------------------------------------------------------
    CUDA_METHOD explicit RotatedRect(const Point& po_1, const Point& po_2,
                                     const FLOAT_TYPE height, const bool rect_up)
    // -----------------------------------------------------------------------------------
    {
        this->hack_flash = false;
        this->rect_up = rect_up;

        this->pD = po_1;
        this->pC = po_2;

        if (std::abs(pD.x-pC.x)<0.0001)
        {
            pD.x += static_cast<FLOAT_TYPE>(0.0001);
            hack_flash = true;
        }
        if (std::abs(pD.y-pC.y)<0.0001)
        {
            pD.y += static_cast<FLOAT_TYPE>(0.0001);
            hack_flash = true;
        }

        this->lCD = Line(this->pC, this->pD);
        this->lBC = Line::perpLinePassPoint(this->lCD, this->pC);
        this->lDA = Line::perpLinePassPoint(this->lCD, this->pD);

        if (this->rect_up)
            this->lAB = Line(this->lCD.M(), lCD.B()-height*std::sqrt(lCD.M()*lCD.M()+1));
        else
            this->lAB = Line(this->lCD.M(), lCD.B()+height*std::sqrt(lCD.M()*lCD.M()+1));

        FLOAT_TYPE xA = (lDA.B()-lAB.B()) / (lAB.M() - lDA.M());
        FLOAT_TYPE xB = (lBC.B()-lAB.B()) / (lAB.M() - lBC.M());
        this->pA = Point(xA,lAB.Y(xA));
        this->pB = Point(xB,lAB.Y(xB));

        this->half_perimeter = static_cast<FLOAT_TYPE>( norm(this->pA-this->pB) + norm(this->pB-this->pC) );
        this->inner_rect_threshold = static_cast<FLOAT_TYPE>(0.01);
    }

    // -----------------------------------------------------------------------------------
    CUDA_METHOD void GetCorners(Point &pA, Point &pB, Point &pC, Point &pD) const
    // -----------------------------------------------------------------------------------
    {
        pA = this->pA;
        pB = this->pB;
        pC = this->pC;
        pD = this->pD;
    }

    // -----------------------------------------------------------------------------------
    CUDA_METHOD void GetLines(Line &lAB, Line &lBC, Line &lCD, Line &lDA) const
    // -----------------------------------------------------------------------------------
    {
        lAB = this->lAB;
        lBC = this->lBC;
        lCD = this->lCD;
        lDA = this->lDA;
    }

    // -----------------------------------------------------------------------------------
    CUDA_METHOD FLOAT_TYPE Area() const
    // -----------------------------------------------------------------------------------
    {
        return norm(this->pA-this->pB)*norm(this->pB-this->pC);
    }

    // -----------------------------------------------------------------------------------
    CUDA_METHOD Point GetCentralPoint() const
    // -----------------------------------------------------------------------------------
    {
        return Point( (pA.x+pB.x+pC.x+pD.x)/4.0f, (pA.y+pB.y+pC.y+pD.y)/4.f );
    }

    // -----------------------------------------------------------------------------------
    CUDA_METHOD Point GetExtremalPoint() const
    // -----------------------------------------------------------------------------------
    {
        return Point( (pA.x+pB.x)/2.0f, (pA.y+pB.y)/2.f );
    }

    // -----------------------------------------------------------------------------------
    CUDA_METHOD int FindVectorsSize(const int im_rows, const int im_cols) const
    // -----------------------------------------------------------------------------------
    {
        const FLOAT_TYPE min_x = fmaxf(fminf(fminf(fminf(pA.x, pB.x), pC.x), pD.x), 0.f);
        const FLOAT_TYPE min_y = fmaxf(fminf(fminf(fminf(pA.y, pB.y), pC.y), pD.y), 0.f);
        const FLOAT_TYPE max_x = fminf(fmaxf(fmaxf(fmaxf(pA.x, pB.x), pC.x), pD.x), static_cast<FLOAT_TYPE>(im_cols));
        const FLOAT_TYPE max_y = fminf(fmaxf(fmaxf(fmaxf(pA.y, pB.y), pC.y), pD.y), static_cast<FLOAT_TYPE>(im_rows));

        int size_of_vectors = 0;

        for(int y = static_cast<int>(min_y); y < static_cast<int>(max_y); ++y)
        {
            for(int x = static_cast<int>(min_x); x < static_cast<int>(max_x); ++x)
            {
                if (!hack_flash)
                {
                    const FLOAT_TYPE d_ab = fabsf(-lAB.M() * x + y - lAB.B()) / sqrtf(1 + lAB.M() * lAB.M());
                    const FLOAT_TYPE d_bc = fabsf(-lBC.M() * x + y - lBC.B()) / sqrtf(1 + lBC.M() * lBC.M());
                    const FLOAT_TYPE d_da = fabsf(-lDA.M() * x + y - lDA.B()) / sqrtf(1 + lDA.M() * lDA.M());
                    const FLOAT_TYPE d_cd = fabsf(-lCD.M() * x + y - lCD.B()) / sqrtf(1 + lCD.M() * lCD.M());

                    if (fabsf((d_ab + d_bc + d_cd + d_da) - half_perimeter ) <= inner_rect_threshold)
                        size_of_vectors++;
                }
                else
                    size_of_vectors++;
            }
        }

        return size_of_vectors;
    }

    // -----------------------------------------------------------------------------------
    CUDA_METHOD void FillVectors(const int im_rows, const int im_cols,
                                FLOAT_TYPE *distances, Point *points, size_t vectors_size) const
    // -----------------------------------------------------------------------------------
    {
        const FLOAT_TYPE min_x = fmax(fmin(fmin(fmin(pA.x, pB.x), pC.x), pD.x), 0.f);
        const FLOAT_TYPE min_y = fmax(fmin(fmin(fmin(pA.y, pB.y), pC.y), pD.y), 0.f);
        const FLOAT_TYPE max_x = fmin(fmax(fmax(fmax(pA.x, pB.x), pC.x), pD.x), static_cast<FLOAT_TYPE>(im_cols));
        const FLOAT_TYPE max_y = fmin(fmax(fmax(fmax(pA.y, pB.y), pC.y), pD.y), static_cast<FLOAT_TYPE>(im_rows));

        int cont=0;
        for(int y = static_cast<int>(min_y); y < static_cast<int>(max_y); ++y)
        {
            for(int x = static_cast<int>(min_x); x < static_cast<int>(max_x) && cont < vectors_size; ++x)
            {
                const FLOAT_TYPE d_cd = fabs(-lCD.M() * x + y - lCD.B()) / sqrtf(1 + lCD.M() * lCD.M());

                if (!hack_flash)
                {
                    const FLOAT_TYPE d_ab = fabs(-lAB.M() * x + y - lAB.B()) / sqrtf(1 + lAB.M() * lAB.M());
                    const FLOAT_TYPE d_bc = fabs(-lBC.M() * x + y - lBC.B()) / sqrtf(1 + lBC.M() * lBC.M());
                    const FLOAT_TYPE d_da = fabs(-lDA.M() * x + y - lDA.B()) / sqrtf(1 + lDA.M() * lDA.M());

                    if (fabs((d_ab + d_bc + d_cd + d_da) - half_perimeter ) <= inner_rect_threshold)
                    {
                        distances[cont] = d_cd;
                        points[cont] = Point(static_cast<FLOAT_TYPE>(x), static_cast<FLOAT_TYPE>(y));
                        ++cont;
                    }
                }
                else
                {
                    distances[cont] = d_cd;
                    points[cont] = Point(static_cast<FLOAT_TYPE>(x), static_cast<FLOAT_TYPE>(y));
                    ++cont;
                }
            }
        }
    }

public:
    Point pA, pB, pC, pD;
    Line lAB, lBC, lCD, lDA;
    bool rect_up;
    FLOAT_TYPE half_perimeter;
    bool hack_flash;
    FLOAT_TYPE inner_rect_threshold;
};

/*!
 * \brief cuda_compute_costs All the cuda magic
 * \param mat_b_values
 * \param p_c_x_values
 * \param points_a
 * \param points_b
 * \param costs
 * \param height
 */

// -----------------------------------------------------------------------------------
void cuda_compute_all_costs(const std::vector<std::vector<FLOAT_TYPE> >& mat_b_values,
                            const std::vector<std::vector<FLOAT_TYPE> >& p_c_x_values,
                            const std::vector<bool>& flags_imp,
                            const std::vector<cuda_roam::Point>& window_tls,
                            const std::vector<cuda_roam::Point>& window_sizes,
                            const std::vector<Point>& points_a, const std::vector<Point>& points_b,
                            std::vector<FLOAT_TYPE>& costs, const FLOAT_TYPE height,
                            const FLOAT_TYPE sigma_color, const int number_nodes,
                            const int rows_image, const int cols_image, const int label_space_size,
                            const FLOAT_TYPE weight);
// -----------------------------------------------------------------------------------

} // namespace cuda_roam

