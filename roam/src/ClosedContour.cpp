#include "ClosedContour.h"

using namespace ROAM;

//-----------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------- ClosedContour ------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
ClosedContour::ClosedContour(const ClosedContour::Params &parameters, bool foreground_inside)
// -----------------------------------------------------------------------------------
{
    this->foreground_inside = foreground_inside;
    params = parameters;
    dp_table_built = false;
}

// -----------------------------------------------------------------------------------
void ClosedContour::BuildDPTable()
// -----------------------------------------------------------------------------------
{
    const uint number_nodes = static_cast<uint>(contour_nodes.size());

    if(costs.size() > 0)
    {
        assert(dp_table_built);

        // ASSUMES DP_TABLE WAS PREVIOUSLY INITIALIZED!!!
        const size_t max_num_labels_sq = this->params.max_number_labels * this->params.max_number_labels;

        #pragma omp parallel for
        for(auto n = 0; n < costs.size(); ++n)
        {
            const size_t ind_node = n / max_num_labels_sq;
            const size_t label_span = n % max_num_labels_sq;

            const size_t l1 = label_span / this->params.max_number_labels;
            const size_t l2 = label_span % this->params.max_number_labels;

            dp_table->pairwise_costs[ind_node][l1][l2] += costs[n];
        }
    }
    else
    {
        dp_table = std::make_shared<ClosedChainDPTable>(params.max_number_labels, number_nodes);
        dp_table->Initialize();

        // openMP does not support std::list
        std::vector<Node*> elements;
        for(auto it = contour_nodes.begin(); it != contour_nodes.end(); ++it)
          elements.push_back(&(*it));

        // fill in unary terms
        #pragma omp parallel for
        for(auto n = 0; n < elements.size(); ++n)
            for(auto l = 0; l < static_cast<int>(elements[n]->GetLabelSpaceSize()); ++l)
                dp_table->unary_costs[n][l] = elements[n]->GetTotalUnaryCost(l);

        // fill in pairwise terms
        #pragma omp parallel for
        for(auto n = 0; n < elements.size() - 1; ++n)
            for(auto l1 = 0; l1 < static_cast<int>(elements[n]->GetLabelSpaceSize()); ++l1)
                for(auto l2 = 0; l2 < static_cast<int>(elements[n + 1]->GetLabelSpaceSize()); ++l2)
                    dp_table->pairwise_costs[n][l1][l2] = elements[n]->GetTotalPairwiseCost(static_cast<ROAM::label>(l1), static_cast<ROAM::label>(l2), *elements[n + 1]);

        // fill in the last edge (closed the contour)
        #pragma omp parallel for
        for(auto l1 = 0; l1 < static_cast<int>(elements[number_nodes - 1]->GetLabelSpaceSize()); ++l1)
            for(auto l2 = 0; l2 < static_cast<int>(elements[0]->GetLabelSpaceSize()); ++l2)
                dp_table->pairwise_costs[number_nodes - 1][l1][l2] = elements[number_nodes - 1]->GetTotalPairwiseCost(static_cast<ROAM::label>(l1), static_cast<ROAM::label>(l2), *elements[0]);
    }

    // Table was built
    dp_table_built = true;
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE ClosedContour::RunDPInference()
// -----------------------------------------------------------------------------------
{
    assert(dp_table_built);

    FLOAT_TYPE min_cost = std::numeric_limits<FLOAT_TYPE>::max();
    current_solution = dp_solver.Minimize(this->dp_table, min_cost);

    // TODO: do we really need this?
    costs.clear();

    return min_cost;
}

// -----------------------------------------------------------------------------------
std::vector<label> ClosedContour::GetCurrentSolution() const
// -----------------------------------------------------------------------------------
{
    return this->current_solution;
}

// -----------------------------------------------------------------------------------
void ClosedContour::ApplyMoves()
// -----------------------------------------------------------------------------------
{
    assert(contour_nodes.size()==current_solution.size());

    // TODO: yes, it's a list but should be parallelized as well
    size_t ind_sol = 0;
    for(auto it = contour_nodes.begin(); it != contour_nodes.end(); ++it, ++ind_sol)
        it->SetCoordinates(current_solution[ind_sol]);
}

// -----------------------------------------------------------------------------------
std::shared_ptr<DPTable> ClosedContour::GetDPTable() const
// -----------------------------------------------------------------------------------
{
    return dp_table;
}

// -----------------------------------------------------------------------------------
template<typename T>
static void MatrixToVector(const cv::Mat& matrix, std::vector<FLOAT_TYPE>& mat_values)
// -----------------------------------------------------------------------------------
{
    const size_t n_elements = matrix.rows * matrix.cols;
    mat_values.resize(n_elements);

    #pragma omp parallel for
    for(auto i = 0; i < n_elements; ++i)
        mat_values[i] = static_cast<FLOAT_TYPE>(matrix.at<T>(i)); // TODO: replace .at<T>
}

// -----------------------------------------------------------------------------------
template<typename T>
static void VectorToMatrix(cv::Mat& matrix, const std::vector<FLOAT_TYPE>& mat_values, const int cols)
// -----------------------------------------------------------------------------------
{
    #pragma omp parallel for
    for(auto i = 0; i < mat_values.size(); ++i)
        matrix.at<T>(i / cols, i % cols) = static_cast<T>(mat_values[i]);
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE ClosedContour::GetTotalContourCost(const FLOAT_TYPE win_height,
                                              const FLOAT_TYPE dist_sigma,
                                              const FLOAT_TYPE weight_snapcut,
                                              const int im_rows, 
                                              const int im_cols, 
                                              const bool use_sc) const
// -----------------------------------------------------------------------------------
{
    const uint number_nodes = static_cast<uint>(contour_nodes.size());

    // openMP does not support std::list
    std::vector<const Node*> elements;
    for(auto it = contour_nodes.begin(); it != contour_nodes.end(); ++it)
      elements.push_back(&(*it));

    FLOAT_TYPE cost = 0.0f;

    // TODO: parallelize with reduction
    // unary terms
    for(auto n = 0; n < elements.size(); ++n)
        cost += elements[n]->GetTotalUnaryCost(0);

    for(auto n = 0; n < elements.size() - 1; ++n)
        cost += elements[n]->GetTotalPairwiseCost(0, 0, *elements[n+1]);

    cost += elements[number_nodes-1]->GetTotalPairwiseCost(0, 0, *elements[0]);

    const FLOAT_TYPE cuda_cost = use_sc ? computeContourCudaCost(win_height, dist_sigma, weight_snapcut, im_rows, im_cols) : 0.0f;

    return cost + cuda_cost;
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE ClosedContour::computeContourCudaCost(const FLOAT_TYPE win_height,
                                                 const FLOAT_TYPE dist_sigma,
                                                 const FLOAT_TYPE weight_snapcut,
                                                 const int im_rows, 
                                                 const int im_cols) const
// -----------------------------------------------------------------------------------
{
#ifdef WITH_CUDA
    std::vector<cuda_roam::Point> gmm_window_tl;
    std::vector<std::vector<FLOAT_TYPE>> vector_p_c_x;
    std::vector<std::vector<FLOAT_TYPE>> vector_mat_b;
    std::vector<cuda_roam::Point> window_sizes;
    std::vector<bool> vector_flag_imp;

    std::vector<cuda_roam::Point> coordinates_a;
    std::vector<cuda_roam::Point> coordinates_b;

    coordinates_a.reserve(contour_nodes.size());
    coordinates_b.reserve(contour_nodes.size());

    gmm_window_tl.reserve(contour_nodes.size());
    vector_p_c_x.reserve(contour_nodes.size());
    vector_mat_b.reserve(contour_nodes.size());
    vector_flag_imp.reserve(contour_nodes.size());

    for(auto it = contour_nodes.begin(); it != contour_nodes.end(); ++it)
    {
        std::list<Node>::const_iterator it_next;
        if(it == --contour_nodes.end())
            it_next = contour_nodes.begin();
        else
            it_next = std::next(it,1);

        const cv::Point a = it->GetCoordinates();
        const cv::Point b = it_next->GetCoordinates();
        coordinates_a.push_back(cuda_roam::Point(static_cast<FLOAT_TYPE>(a.x), static_cast<FLOAT_TYPE>(a.y)));
        coordinates_b.push_back(cuda_roam::Point(static_cast<FLOAT_TYPE>(b.x), static_cast<FLOAT_TYPE>(b.y)));

        // TODO: parallelize with binary flag and erase
        for(size_t pw = 0; pw < it->pairwise_terms.size(); ++pw)
        {
            const cv::Ptr<SnapcutPairwise> snapcutPairwise = it->pairwise_terms[pw].dynamicCast<SnapcutPairwise>();
            if(!snapcutPairwise.empty())
            {
                const cv::Mat& mat_b = snapcutPairwise->GetMatrixB();

                cv::Mat evaluated_gmms;
                cv::Rect gmms_window;
                bool edge_initialized;
                bool edge_impossible;

                snapcutPairwise->GetModelElements(evaluated_gmms, gmms_window, edge_impossible, edge_initialized);

                const cv::Mat& sub_matrix_b = mat_b(gmms_window);

                std::vector<FLOAT_TYPE> sub_matb_values, pcx_values;
                MatrixToVector<unsigned char>(sub_matrix_b, sub_matb_values);
                MatrixToVector<FLOAT_TYPE>(evaluated_gmms, pcx_values);

                window_sizes.push_back(cuda_roam::Point(static_cast<FLOAT_TYPE>(gmms_window.width), static_cast<FLOAT_TYPE>(gmms_window.height)));
                gmm_window_tl.push_back(cuda_roam::Point(static_cast<FLOAT_TYPE>(gmms_window.x), static_cast<FLOAT_TYPE>(gmms_window.y)));
                vector_p_c_x.push_back(pcx_values);
                vector_mat_b.push_back(sub_matb_values);
                vector_flag_imp.push_back(edge_impossible);
            }
        }
    }

    // Now, call the CUDA function
    std::vector<FLOAT_TYPE> costs_coordinates; //!< costs computed with cuda
    cuda_roam::cuda_compute_all_costs(vector_mat_b, vector_p_c_x, vector_flag_imp, gmm_window_tl, window_sizes,
                                      coordinates_a, coordinates_b, costs_coordinates, win_height, dist_sigma,
                                      static_cast<int>(contour_nodes.size()), im_rows, im_cols, 1,
                                      weight_snapcut);

    const FLOAT_TYPE cuda_cost = std::accumulate(costs_coordinates.begin(), costs_coordinates.end(), static_cast<FLOAT_TYPE>(0.0));
#else
    const FLOAT_TYPE cuda_cost = 0.f;
#endif

    return cuda_cost;
}

// -----------------------------------------------------------------------------------
void ClosedContour::ExecuteCudaPairwises(const FLOAT_TYPE win_height,
                                         const FLOAT_TYPE dist_sigma,
                                         const FLOAT_TYPE weight_snapcut,
                                         const int im_rows, const int im_cols)
// -----------------------------------------------------------------------------------
{
#ifdef WITH_CUDA	
    // Start with the memory preparation
    std::vector<cuda_roam::Point> gmm_window_tl;
    std::vector<std::vector<FLOAT_TYPE> > vector_p_c_x;
    std::vector<std::vector<FLOAT_TYPE> > vector_mat_b;
    std::vector<cuda_roam::Point> window_sizes;
    std::vector<bool> vector_flag_imp;

    std::vector<cuda_roam::Point> coordinates_a;
    std::vector<cuda_roam::Point> coordinates_b;

    coordinates_a.reserve(this->params.max_number_labels * this->params.max_number_labels * contour_nodes.size());
    coordinates_b.reserve(this->params.max_number_labels * this->params.max_number_labels * contour_nodes.size());

    gmm_window_tl.reserve(contour_nodes.size());
    vector_p_c_x.reserve(contour_nodes.size());
    vector_mat_b.reserve(contour_nodes.size());
    vector_flag_imp.reserve(contour_nodes.size());

    for(auto it = contour_nodes.begin(); it != contour_nodes.end(); ++it)
    {
        std::list<Node>::iterator it_next;
        if(it == --contour_nodes.end())
            it_next = contour_nodes.begin();
        else
            it_next = std::next(it,1);

        for(label ca = 0; ca < this->params.max_number_labels; ++ca)
            for(label cb = 0; cb < this->params.max_number_labels; ++cb)
            {
                const cv::Point a = it->getDisplacedPointFromLabel(ca);
                const cv::Point b = it_next->getDisplacedPointFromLabel(cb);
                coordinates_a.push_back(cuda_roam::Point(static_cast<FLOAT_TYPE>(a.x), static_cast<FLOAT_TYPE>(a.y)));
                coordinates_b.push_back(cuda_roam::Point(static_cast<FLOAT_TYPE>(b.x), static_cast<FLOAT_TYPE>(b.y)));
            }

        // TODO: parallelize with binary flag and erase
        for(size_t pw = 0; pw < it->pairwise_terms.size(); ++pw)
        {
            const cv::Ptr<SnapcutPairwise> snapcutPairwise = it->pairwise_terms[pw].dynamicCast<SnapcutPairwise>();
            if(!snapcutPairwise.empty())
            {
                const cv::Mat &mat_b = snapcutPairwise->GetMatrixB();

                cv::Mat evaluated_gmms;
                cv::Rect gmms_window;
                bool edge_initialized;
                bool edge_impossible;

                snapcutPairwise->GetModelElements(evaluated_gmms, gmms_window, edge_impossible, edge_initialized);

                const cv::Mat& sub_matrix_b = mat_b(gmms_window);

                std::vector<FLOAT_TYPE> sub_matb_values, pcx_values;
                MatrixToVector<unsigned char>(sub_matrix_b, sub_matb_values);
                MatrixToVector<FLOAT_TYPE>(evaluated_gmms, pcx_values);

                window_sizes.push_back(cuda_roam::Point(static_cast<FLOAT_TYPE>(gmms_window.width), static_cast<FLOAT_TYPE>(gmms_window.height)));
                gmm_window_tl.push_back(cuda_roam::Point(static_cast<FLOAT_TYPE>(gmms_window.x), static_cast<FLOAT_TYPE>(gmms_window.y)));
                vector_p_c_x.push_back(pcx_values);
                vector_mat_b.push_back(sub_matb_values);
                vector_flag_imp.push_back(edge_impossible);
            }
        }
    }

    // Now, call the CUDA function
    cuda_roam::cuda_compute_all_costs(vector_mat_b, vector_p_c_x, vector_flag_imp, gmm_window_tl, window_sizes,
                                      coordinates_a, coordinates_b, costs, win_height, dist_sigma,
                                      static_cast<int>(contour_nodes.size()), im_rows, im_cols, this->params.max_number_labels,
                                      weight_snapcut);
#endif
}

// -----------------------------------------------------------------------------------
cv::Mat ClosedContour::GenerateLikelihoodMap(const cv::Mat& image)
// -----------------------------------------------------------------------------------
{
    cv::Mat maps[2];

    maps[0] = cv::Mat(image.size(), CV_32FC1, cv::Scalar(0.f));
    maps[1] = cv::Mat(image.size(), CV_32FC1, cv::Scalar(0.f));    

    for(auto it = contour_nodes.begin(); it != contour_nodes.end(); ++it)
    {
        // TODO: parallelize with binary flag and erase
        for(size_t pw=0; pw < it->pairwise_terms.size(); ++pw)
        {
            const cv::Ptr<SnapcutPairwise> &snapcutPairwise = it->pairwise_terms[pw].dynamicCast<SnapcutPairwise>();
            if(!snapcutPairwise.empty())
            {
                cv::Rect window;
                GMMModel fg, bg;
                bool is_edge_imp;
                cv::Mat fg_eval, bg_eval;
                snapcutPairwise->GetModels(fg, bg, window, is_edge_imp, fg_eval, bg_eval);

                if(!is_edge_imp)
                {
                    maps[0](window) += fg_eval;
                    maps[1](window) += bg_eval;
                }
            }
        }
    }

    cv::Mat map;
    cv::merge(maps, 2, map);
    return map;
}

// -----------------------------------------------------------------------------------
void ClosedContour::SetForegroundSideFlag(const bool flag)
// -----------------------------------------------------------------------------------
{
    this->foreground_inside = flag;
}

// -----------------------------------------------------------------------------------
bool ClosedContour::GetForegroundSideFlag() const
// -----------------------------------------------------------------------------------
{
    return this->foreground_inside;
}

// -----------------------------------------------------------------------------------
cv::Mat ClosedContour::DrawContour(const cv::Mat &image, const cv::Mat &mask, bool draw_nodes) const
// -----------------------------------------------------------------------------------
{
    cv::Mat output = image.clone();

    if(!mask.empty())
    {
        output /= 2;
        image.copyTo(output, mask);
    }

    for(auto it = this->contour_nodes.begin(); it != this->contour_nodes.end(); ++it)
    {
        const cv::Point n_c = it->GetCoordinates();

        if(it == --this->contour_nodes.end())
            cv::line(output, it->GetCoordinates(), this->contour_nodes.begin()->GetCoordinates(),cv::Scalar(0,255,255),7);
        else
            cv::line(output, it->GetCoordinates(), std::next(it,1)->GetCoordinates(),cv::Scalar(0,255,255),7,cv::LINE_AA);

        if(draw_nodes)
           cv::circle(output, n_c, 4, cv::Scalar(0,200,255), -1);
    }

    return output;
}

// -----------------------------------------------------------------------------------
void ClosedContour::PruneNodes(const std::vector<bool> &toRemove)
// -----------------------------------------------------------------------------------
{
    assert(toRemove.size() == contour_nodes.size());

    // TODO: parallelize (yes, I know it's a list)
    int i = 0;
    for(auto it = contour_nodes.begin(); it != contour_nodes.end(); ++it, ++i)
        it->remove = toRemove[i];

    contour_nodes.erase(std::remove_if(contour_nodes.begin(), contour_nodes.end(), [](const Node & o){ return o.remove;}), contour_nodes.end());
}

// -----------------------------------------------------------------------------------
size_t ClosedContour::AddNode(const cv::Point &pt, const int prev_node)
// -----------------------------------------------------------------------------------
{
    if(prev_node < 0)
    {
        contour_nodes.push_back(Node(pt));
        return contour_nodes.size();
    }
    else
    {
        contour_nodes.insert(std::next(contour_nodes.begin(), prev_node + 1), Node(pt));
        return prev_node + 1;
    }
}

// -----------------------------------------------------------------------------------
std::list<Node>::iterator ClosedContour::AddNode(const cv::Point &pt, std::list<Node>::iterator it)
// -----------------------------------------------------------------------------------
{
    contour_nodes.insert(it, Node(pt));
    return --it;
}

// -----------------------------------------------------------------------------------
bool ClosedContour::IsCounterClockWise() const
// -----------------------------------------------------------------------------------
{
    const size_t stop = contour_nodes.size() - 1;
    FLOAT_TYPE area = 0.0;

    // TODO: again, mae it parallelized (despite it's a list)
    int i=0;
    for(auto it = contour_nodes.begin(); i < stop && it != --contour_nodes.end(); ++it, ++i)
    {
        const cv::Point a = it->GetCoordinates();
        const cv::Point b = std::next(it, 1)->GetCoordinates();
        area += a.x * b.y; // x_i * y_{i+1}
        area -= b.x * a.y; // x_{i+1} * y_i
    }

    // first points
    const cv::Point a = contour_nodes.rbegin()->GetCoordinates();
    const cv::Point b = contour_nodes.begin()->GetCoordinates();
    area += a.x * b.y;
    area -= b.x * a.y;

    // last step (not necesarry)
    area /= 2.0;

    // the y axis is swapped -> clockwise / counterclockwise as well
    const bool counterclockwise = area < 0.0 ? true : false;

    return counterclockwise;
}
