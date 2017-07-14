//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include <cstdlib>

#include "roam_cuda.h"


namespace  cuda_roam
{

// -----------------------------------------------------------------------------------
CUDA_METHOD bool GetElements(const int im_rows, const int im_cols, const RotatedRect& rect,
                                 FLOAT_TYPE &distance, Point &point, size_t index)
// -----------------------------------------------------------------------------------
{
    const FLOAT_TYPE min_x = fmaxf(fminf(fminf(fminf(rect.pA.x, rect.pB.x), rect.pC.x), rect.pD.x), 0.f);
    const FLOAT_TYPE min_y = fmaxf(fminf(fminf(fminf(rect.pA.y, rect.pB.y), rect.pC.y), rect.pD.y), 0.f);
    const FLOAT_TYPE max_x = fminf(fmaxf(fmaxf(fmaxf(rect.pA.x, rect.pB.x), rect.pC.x), rect.pD.x), static_cast<FLOAT_TYPE>(im_cols));
    const FLOAT_TYPE max_y = fminf(fmaxf(fmaxf(fmaxf(rect.pA.y, rect.pB.y), rect.pC.y), rect.pD.y), static_cast<FLOAT_TYPE>(im_rows));

    int bound_rect_cols = static_cast<int>(max_x - min_x);
    int bound_rect_rows = static_cast<int>(max_y - min_y);

    if ((int)index > bound_rect_rows*bound_rect_cols)
        return false;


    int y = static_cast<int>(min_y) + (int)index/bound_rect_cols;
    int x = static_cast<int>(min_x) + (int)index%bound_rect_cols;

    const FLOAT_TYPE d_cd = fabsf(-rect.lCD.M() * (FLOAT_TYPE)x + (FLOAT_TYPE)y - rect.lCD.B()) / sqrtf(1.f + rect.lCD.M() * rect.lCD.M());

    if (!rect.hack_flash)
    {
        const FLOAT_TYPE d_ab = fabsf(-rect.lAB.M() * (FLOAT_TYPE)x + (FLOAT_TYPE)y - rect.lAB.B()) / sqrtf(1.f + rect.lAB.M() * rect.lAB.M());
        const FLOAT_TYPE d_bc = fabsf(-rect.lBC.M() * (FLOAT_TYPE)x + (FLOAT_TYPE)y - rect.lBC.B()) / sqrtf(1.f + rect.lBC.M() * rect.lBC.M());
        const FLOAT_TYPE d_da = fabsf(-rect.lDA.M() * (FLOAT_TYPE)x + (FLOAT_TYPE)y - rect.lDA.B()) / sqrtf(1.f + rect.lDA.M() * rect.lDA.M());

        if (fabsf((d_ab + d_bc + d_cd + d_da) - rect.half_perimeter ) <= rect.inner_rect_threshold)
        {
            distance = d_cd;
            point = Point(static_cast<FLOAT_TYPE>(x), static_cast<FLOAT_TYPE>(y));
            return true;
        }
        else
            return false;
    }
    else
    {
        distance = d_cd;
        point = Point(static_cast<FLOAT_TYPE>(x), static_cast<FLOAT_TYPE>(y));
        return true;
    }
}

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define MAX_MEM 1000
#define MAX_ITERS 2000


struct all_costs_functor
{

    const int image_rows;
    const int image_cols;
    const int num_nodes;
    const int num_labels;

    const thrust::device_ptr<FLOAT_TYPE> d_mat_b_values;
    const thrust::device_ptr<FLOAT_TYPE> d_p_c_x_values;

    const thrust::device_ptr<Point> d_win_tls;
    const thrust::device_ptr<Point> d_win_szs;

    const thrust::device_ptr<bool> d_flags_imp;

    const FLOAT_TYPE height;
    const FLOAT_TYPE sigma_color;
    const FLOAT_TYPE cost_impossible;
    const FLOAT_TYPE w_snapcut;

    all_costs_functor
    (
         const thrust::device_ptr<FLOAT_TYPE> _mat_b_values,
         const thrust::device_ptr<FLOAT_TYPE> _p_c_x_values,
         const thrust::device_ptr<bool> _flags_imp,
         const thrust::device_ptr<Point> _d_win_tls,
         const thrust::device_ptr<Point> _d_win_szs,
         const FLOAT_TYPE _height,
         const FLOAT_TYPE _sigma_color,
         const int _image_rows, const int _image_cols,
         const int _num_nodes, const int _num_labels,
         const FLOAT_TYPE _cost_impossible, const FLOAT_TYPE _w_sc
    ) : image_rows(_image_rows),
        image_cols(_image_cols),
        num_nodes(_num_nodes),
        num_labels(_num_labels),
        d_mat_b_values(_mat_b_values),
        d_p_c_x_values(_p_c_x_values), d_win_tls(_d_win_tls),
        d_win_szs(_d_win_szs), d_flags_imp(_flags_imp),
        height(_height), sigma_color(_sigma_color),
        cost_impossible(_cost_impossible), w_snapcut(_w_sc)
    {}

    template <typename T>
    CUDA_METHOD
    void operator()(T tuple) const
    {
        const Point &c_a = thrust::get<0>(tuple);
        const Point &c_b = thrust::get<1>(tuple);

        const int &index = thrust::get<2>(tuple);

        const RotatedRect rect_1(c_a, c_b, height, true );
        const RotatedRect rect_2(c_a, c_b, height, false);

        // Find out p_c_x interval of this thread
        int num_points_per_edge = num_labels*num_labels;
        int index_edge = index/num_points_per_edge;

        Point siz_window = d_win_szs[index_edge];
        Point tls_window = d_win_tls[index_edge];
        bool is_impossible_edge = d_flags_imp[index_edge];

        if (is_impossible_edge)
        {
            if (fabsf(c_a.x-c_b.x)<2.f && fabsf(c_a.y-c_b.y)<2.f)
                thrust::get<3>(tuple) = 1.0f * w_snapcut;
            else
                thrust::get<3>(tuple) = 0.0f;

            return;
        }

        size_t index_mats_ini = 0;
        for (int i=0; i<index_edge; ++i)
        {
            const Point& sz_i = d_win_szs[i];
            index_mats_ini += static_cast<size_t>(sz_i.x*sz_i.y);
        }
        size_t index_mats_end = index_mats_ini + static_cast<size_t>(siz_window.x * siz_window.y);

        Point r1_cp = rect_1.GetExtremalPoint();
        r1_cp.x -= tls_window.x;
        r1_cp.y -= tls_window.y;

        Point r2_cp = rect_2.GetExtremalPoint();
        r2_cp.x -= tls_window.x;
        r2_cp.y -= tls_window.y;


        FLOAT_TYPE fcp1, fcp2;
        size_t index_fcp1 = index_mats_ini + static_cast<size_t>(r1_cp.y*siz_window.x + r1_cp.x);
        size_t index_fcp2 = index_mats_ini + static_cast<size_t>(r2_cp.y*siz_window.x + r2_cp.x);

        if (index_fcp1<index_mats_end && index_fcp1>=index_mats_ini)
            fcp1 = d_mat_b_values[ index_fcp1 ];
        else
        {
            thrust::get<3>(tuple) = w_snapcut;
            return;
        }

        if (index_fcp2<index_mats_end && index_fcp2>=index_mats_ini)
            fcp2 = d_mat_b_values[ index_fcp2 ];
        else
        {
            thrust::get<3>(tuple) = w_snapcut;
            return;
        }

        int fg_n_points, bg_n_points;

        if (fcp1>=fcp2)
        {
            fg_n_points = min(rect_1.FindVectorsSize(image_rows, image_cols), MAX_MEM);
            bg_n_points = min(rect_2.FindVectorsSize(image_rows, image_cols), MAX_MEM);
        }
        else
        {
            fg_n_points = min(rect_2.FindVectorsSize(image_rows, image_cols), MAX_MEM);
            bg_n_points = min(rect_1.FindVectorsSize(image_rows, image_cols), MAX_MEM);
        }

        FLOAT_TYPE normalizer = 0.f;
        FLOAT_TYPE fc = 0;

        int num_fg_val_elems = 0;
        int index_fg = 0;
        while(num_fg_val_elems<fg_n_points && index_fg<MAX_ITERS)
        {
            Point pt;
            FLOAT_TYPE dis;
            bool valid_el;

            if (fcp1>=fcp2)
                valid_el = GetElements(image_rows, image_cols, rect_1, dis, pt, index_fg++);
            else
                valid_el = GetElements(image_rows, image_cols, rect_2, dis, pt, index_fg++);

            if (valid_el)
                ++num_fg_val_elems;
            else
                continue;

            const int x = static_cast<int>(pt.x - tls_window.x);
            const int y = static_cast<int>(pt.y - tls_window.y);

            if (x>=siz_window.x || y>=siz_window.y || x<0 || y<0)
                continue;

            const FLOAT_TYPE w_c_x = 1.f-
                    expf( -dis*dis/(sigma_color*sigma_color) );

            size_t index_pcx = static_cast<size_t>(index_mats_ini + siz_window.x*y + x);

            if (index_pcx>=index_mats_end)
                continue;

            fc += 1.f - (d_p_c_x_values[index_pcx])*w_c_x;
            normalizer += w_c_x;
        }

        int num_bg_val_elems = 0;
        int index_bg = 0;
        while(num_bg_val_elems<bg_n_points && index_bg<MAX_ITERS)
        {
            Point pt;
            FLOAT_TYPE dis;
            bool valid_el;

            if (fcp1>=fcp2)
                valid_el = GetElements(image_rows, image_cols, rect_2, dis, pt, index_bg++);
            else
                valid_el = GetElements(image_rows, image_cols, rect_1, dis, pt, index_bg++);

            if (valid_el)
                ++num_bg_val_elems;
            else
                continue;

            const int x = static_cast<int>(pt.x - tls_window.x);
            const int y = static_cast<int>(pt.y - tls_window.y);

            if (x>=siz_window.x || y>=siz_window.y || x<0 || y<0)
                continue;

            const FLOAT_TYPE w_c_x = 1.f-
                    expf( -dis*dis/(sigma_color*sigma_color) );

            size_t index_pcx = static_cast<size_t>(index_mats_ini + siz_window.x*y + x);

            if (index_pcx>=index_mats_end)
                continue;

            fc += (d_p_c_x_values[index_pcx])*w_c_x;
            normalizer += w_c_x;
        }

        thrust::get<3>(tuple) = static_cast<FLOAT_TYPE>(w_snapcut * fc / (normalizer + 1e-15));

    }
};

// -----------------------------------------------------------------------------------
void cuda_compute_all_costs(const std::vector<std::vector<FLOAT_TYPE> >& mat_b_values,
                            const std::vector<std::vector<FLOAT_TYPE> >& p_c_x_values, const std::vector<bool> &flags_imp,
                            const std::vector<cuda_roam::Point>& window_tls,
                            const std::vector<cuda_roam::Point>& window_sizes,
                            const std::vector<Point>& points_a, const std::vector<Point>& points_b,
                            std::vector<FLOAT_TYPE>& costs, const FLOAT_TYPE height,
                            const FLOAT_TYPE sigma_color, const int number_nodes,
                            const int rows_image, const int cols_image, const int label_space_size,
                            const FLOAT_TYPE weight)
// -----------------------------------------------------------------------------------
{
    // Upload from CPU to GPU
    thrust::device_vector<Point> d_points_a = points_a;
    thrust::device_vector<Point> d_points_b = points_b;

    thrust::device_vector<FLOAT_TYPE> d_mat_b_values;
    thrust::device_vector<FLOAT_TYPE> d_p_c_x_values;

    for (size_t i=0; i<mat_b_values.size(); ++i)
        d_mat_b_values.insert(d_mat_b_values.end(), mat_b_values[i].begin(), mat_b_values[i].end());

    for (size_t i=0; i<p_c_x_values.size(); ++i)
        d_p_c_x_values.insert(d_p_c_x_values.end(), p_c_x_values[i].begin(), p_c_x_values[i].end());

    thrust::device_vector<bool> d_flags_imp = flags_imp;
    thrust::device_vector<Point> d_win_tls = window_tls;
    thrust::device_vector<Point> d_win_szs = window_sizes;

    // Make space to fill costs
    costs.resize(points_a.size());
    thrust::device_vector<FLOAT_TYPE> d_costs(points_a.size());

    thrust::counting_iterator<int> init_cont(0);
    thrust::counting_iterator<int> end_cont = init_cont + d_points_a.size();

    // Populate cost: Thrust handles the number of launched kernels
    thrust::for_each(thrust::device,
                thrust::make_zip_iterator(thrust::make_tuple(d_points_a.begin(), d_points_b.begin(), init_cont, d_costs.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(d_points_a.end(), d_points_b.end(), end_cont, d_costs.end())),
                all_costs_functor
                     (
                         d_mat_b_values.data(), d_p_c_x_values.data(), d_flags_imp.data(),
                         d_win_tls.data(), d_win_szs.data(), height, sigma_color,
                         rows_image, cols_image, number_nodes, label_space_size, 1.f, weight
                     )
                    );



    // Download from GPU to CPU
    thrust::copy(d_costs.begin(), d_costs.end(), costs.begin());
}

} // namespace cuda_roam
