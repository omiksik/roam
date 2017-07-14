#include "SnapcutTerms.h"

//-----------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------- DEFINITIONS----------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------

#undef BOILERPLATE_CODE_UNARY
#define BOILERPLATE_CODE_UNARY(name,classname)\
  if(unaryType==name){\
      return classname::createUnaryTerm();\
  }

#undef BOILERPLATE_CODE_PAIRWISE
#define BOILERPLATE_CODE_PAIRWISE(name,classname)\
  if(pairwiseType==name){\
      return classname::createPairwiseTerm();\
  }

using namespace ROAM;

// -----------------------------------------------------------------------------------
const cv::Mat& SnapcutPairwise::GetMatrixA() const
// -----------------------------------------------------------------------------------
{
    return matrix_a;
}

// -----------------------------------------------------------------------------------
const cv::Mat& SnapcutPairwise::GetMatrixB() const
// -----------------------------------------------------------------------------------
{
    return matrix_b;
}

//-----------------------------------------------------------------------------------------------------------------------
//----------------------------------------------- Some static functions -------------------------------------------------

// -----------------------------------------------------------------------------------
static void MatToVector(const cv::Mat& in, std::vector<float>& out)
// -----------------------------------------------------------------------------------
{
    if(in.isContinuous())
        out.assign(reinterpret_cast<const FLOAT_TYPE*>(in.datastart), reinterpret_cast<const FLOAT_TYPE*>(in.dataend));
    else
        for(auto i = 0; i < in.rows; ++i)
            out.insert(out.end(), in.ptr<FLOAT_TYPE>(i), in.ptr<FLOAT_TYPE>(i) + in.cols);
}

//-----------------------------------------------------------------------------------------------------------------------
//------------------------------------- Another Implementation of SnapcutPairwise ---------------------------------------
//-----------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
class SnapcutPairwiseImpl_temporalGMM : public SnapcutPairwise
// -----------------------------------------------------------------------------------
{
public:
    explicit SnapcutPairwiseImpl_temporalGMM(const SnapcutPairwise::Params &parameters = SnapcutPairwise::Params());

    FLOAT_TYPE GetCost(const cv::Point &coordinates_a, const cv::Point &coordinates_b) override;
    FLOAT_TYPE GetWeight() const override;

    bool InitializeEdge(const cv::Point &coordinates_a, const cv::Point &coordinates_b,
                        const cv::Mat &reference_image, const cv::Mat &reference_mask,
                        const cv::Point& diff_a, const cv::Point& diff_b) override;

    void GetModelElements(cv::Mat& evaluated_gmms, cv::Rect& valid_enclosing_rect,
                          bool& is_edge_imp, bool& is_edge_init) override;

    void GetModels(GMMModel &fg_model, GMMModel &bg_model, cv::Rect &win, bool &is_edge_imp,
                   cv::Mat &fg_eval, cv::Mat &bg_eval) override;

protected:

    bool initImpl(const cv::Mat& color_image, const cv::Mat& segmentation_mask) override;
    bool updateImpl(const cv::Mat& mat_a, const cv::Mat& mat_b) override;

private:
    GMMModel fg_gmm;
    GMMModel bg_gmm;
    cv::Mat fg_eval, bg_eval;
    cv::Mat p_c_x; //>
    bool edge_initialized;
    bool edge_impossible;
    cv::Rect valid_enclosing_cvrect;
    LineIntegralImage integral_mask;
};

// -----------------------------------------------------------------------------------
SnapcutPairwiseImpl_temporalGMM::SnapcutPairwiseImpl_temporalGMM(const SnapcutPairwise::Params &parameters)
// -----------------------------------------------------------------------------------
{
    this->is_initialized = false;
    this->matrix_a = cv::Mat();
    this->matrix_b = cv::Mat();
    this->params = parameters;
    this->edge_initialized = false;
    this->edge_impossible = false;
    this->p_c_x = cv::Mat();
    this->valid_enclosing_cvrect = cv::Rect();
    this->fg_gmm = GMMModel(GMMModel::Parameters(this->params.number_clusters));
    this->bg_gmm = GMMModel(GMMModel::Parameters(this->params.number_clusters));
    this->integral_mask = LineIntegralImage();
}

// -----------------------------------------------------------------------------------
bool SnapcutPairwiseImpl_temporalGMM::initImpl(const cv::Mat& color_image,
                                               const cv::Mat& segmentation_mask)
// -----------------------------------------------------------------------------------
{
    if (color_image.size() == segmentation_mask.size())
    {
        this->matrix_a = color_image;
        this->matrix_b = segmentation_mask;
        if(!params.block_get_cost)
            integral_mask = LineIntegralImage::CreateFromImage(this->matrix_b);
        return true;
    }
        
    return false;
}

// -----------------------------------------------------------------------------------
bool SnapcutPairwiseImpl_temporalGMM::updateImpl(const cv::Mat& color_image,
                                                 const cv::Mat& segmentation_mask)
// -----------------------------------------------------------------------------------
{
    if (color_image.size()==segmentation_mask.size())
    {
        this->matrix_a = color_image;
        this->matrix_b = segmentation_mask;
        if(!params.block_get_cost)
            integral_mask = LineIntegralImage::CreateFromImage(this->matrix_b);
        return true;
    }
    
    return false;
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE SnapcutPairwiseImpl_temporalGMM::GetWeight() const
// -----------------------------------------------------------------------------------
{
    return this->params.weight_term;
}

// -----------------------------------------------------------------------------------
void SnapcutPairwiseImpl_temporalGMM::GetModelElements(cv::Mat &evaluated_gmms, 
                                                       cv::Rect &valid_enclosing_rect,
                                                       bool &is_edge_imp, 
                                                       bool &is_edge_init)
// -----------------------------------------------------------------------------------
{
    evaluated_gmms = this->p_c_x;
    valid_enclosing_rect = this->valid_enclosing_cvrect;
    is_edge_imp = this->edge_impossible;
    is_edge_init = this->is_initialized;
}

// -----------------------------------------------------------------------------------
void SnapcutPairwiseImpl_temporalGMM::GetModels(GMMModel &fg_model, 
                                                GMMModel &bg_model,
                                                cv::Rect &win, bool &is_edge_imp,
                                                cv::Mat &fg_eval, cv::Mat &bg_eval)
// -----------------------------------------------------------------------------------
{
    fg_model = this->fg_gmm;
    bg_model = this->bg_gmm;
    win = this->valid_enclosing_cvrect;
    is_edge_imp = this->edge_impossible;
    fg_eval = this->fg_eval;
    bg_eval = this->bg_eval;
}

// -----------------------------------------------------------------------------------
bool SnapcutPairwiseImpl_temporalGMM::InitializeEdge(const cv::Point &coordinates_a,
                                                     const cv::Point &coordinates_b,
                                                     const cv::Mat &reference_image,
                                                     const cv::Mat &reference_mask,
                                                     const cv::Point &diff_a,
                                                     const cv::Point &diff_b)
// -----------------------------------------------------------------------------------
{
    // diff_a and diff_b contain the displacement from cordinates_a to intermediate_motion

    // reset per-edge flags
    this->edge_impossible = false;

    if (!is_initialized )
    {
        this->edge_initialized = false;
        return false;
    }

    if (coordinates_a.x==coordinates_b.x && coordinates_a.y==coordinates_b.y)
    {
        this->edge_initialized = true;
        this->edge_impossible = true;
        return true;
    }

    // Valid_enclosing_cvrect contains all possible pairs of RotatedRects in valid image space
    const FLOAT_TYPE distance_nodes = static_cast<FLOAT_TYPE>(cv::norm(coordinates_a - coordinates_b));
    const FLOAT_TYPE containing_rect_diagonal = std::sqrt(std::pow(distance_nodes + this->params.node_side_length, 2)
                                                    + std::pow(this->params.region_height + this->params.node_side_length, 2));

    const FLOAT_TYPE border_suppl = std::max(containing_rect_diagonal, 20.0f);
    const cv::Point central = (coordinates_a + coordinates_b) / 2;
    const cv::Point top_left = central - cv::Point(static_cast<int>(border_suppl / 2.f), static_cast<int>(border_suppl / 2.f));
    const cv::Rect enclosing_cvrect_ref_frame = cv::Rect(top_left, cv::Size(static_cast<int>(border_suppl), static_cast<int>(border_suppl)));

    const cv::Rect im_rect(0, 0, matrix_a.cols, matrix_a.rows);
    const cv::Rect valid_enclosing_cvrect_ref_frame = im_rect & enclosing_cvrect_ref_frame;

    if(valid_enclosing_cvrect_ref_frame.area() <= 0)
    {
        this->edge_initialized = true;
        this->edge_impossible = true;
        return true;
    }

    // take valid enclosing rect of reference_image (matrix_a if referece_image is empty) and matrix_b
    // if GMM was initialized already, update it instead...
    cv::Mat valid_fmask;
    cv::Mat valid_bmask;

    if(reference_mask.empty())
    {
        cv::erode(this->matrix_b(valid_enclosing_cvrect_ref_frame), valid_fmask, cv::Mat());
        cv::dilate(this->matrix_b(valid_enclosing_cvrect_ref_frame), valid_bmask, cv::Mat());
    }
    else
    {
        cv::erode(reference_mask(valid_enclosing_cvrect_ref_frame), valid_fmask, cv::Mat());
        cv::dilate(reference_mask(valid_enclosing_cvrect_ref_frame), valid_bmask, cv::Mat());
    }

    cv::Mat valid_image_ref;

    if(reference_image.empty())
        valid_image_ref = this->matrix_a(valid_enclosing_cvrect_ref_frame);
    else
        valid_image_ref = reference_image(valid_enclosing_cvrect_ref_frame);

    if(!fg_gmm.initialized)
    {
        //this->edge_impossible = static_cast<bool>(std::abs(fg_gmm.initializeMixture(valid_image_ref, valid_fmask)));
        //this->edge_impossible = static_cast<bool>(std::abs(bg_gmm.initializeMixture(valid_image_ref, 255 - valid_bmask))) || edge_impossible;
        edge_impossible = std::abs(fg_gmm.initializeMixture(valid_image_ref, valid_fmask)) > 0.0 ? true : false;
        edge_impossible = (edge_impossible || std::abs(bg_gmm.initializeMixture(valid_image_ref, 255 - valid_bmask)) > 0.0) ? true : false;
    }
    else
    {
        //this->edge_impossible = static_cast<bool>(std::abs(fg_gmm.updateMixture(valid_image_ref, valid_fmask)));
        //this->edge_impossible = static_cast<bool>(std::abs(bg_gmm.updateMixture(valid_image_ref, 255 - valid_bmask, fg_gmm.getGMMComponents()))) || edge_impossible;
        edge_impossible = std::abs(fg_gmm.updateMixture(valid_image_ref, valid_fmask)) > 0.0 ? true : false;
        edge_impossible = (edge_impossible || std::abs(bg_gmm.updateMixture(valid_image_ref, 255 - valid_bmask, fg_gmm.getGMMComponents())) > 0.0) ? true : false;
    }

    if(this->edge_impossible)
        return true;


    // This should be evaluated in the next image: Correct.
    const cv::Point next_a = coordinates_a + diff_a;
    const cv::Point next_b = coordinates_b + diff_b;
    const cv::Rect enclosing_cvrect_next_frame(std::min(next_a.x, next_b.x) - static_cast<int>(border_suppl),
                                               std::min(next_a.y, next_b.y) - static_cast<int>(border_suppl),
                                               std::abs(next_a.x - next_b.x) + 2 * static_cast<int>(border_suppl),
                                               std::abs(next_a.y - next_b.y) + 2 * static_cast<int>(border_suppl));

    const cv::Rect valid_enclosing_cvrect_next_frame = cv::Rect(0,0,matrix_a.cols,matrix_a.rows) & enclosing_cvrect_next_frame;
    this->valid_enclosing_cvrect = valid_enclosing_cvrect_next_frame;

    this->fg_eval = fg_gmm.getLikelihood(this->matrix_a(valid_enclosing_cvrect_next_frame));
    this->bg_eval = bg_gmm.getLikelihood(this->matrix_a(valid_enclosing_cvrect_next_frame));

    const cv::Mat &fg_likelihood = this->fg_eval;
    const cv::Mat &bg_likelihood = this->bg_eval;

    cv::Mat valid_image_next;
    if (reference_image.empty())
        valid_image_next = this->matrix_a(valid_enclosing_cvrect_next_frame);
    else
        valid_image_next = reference_image(valid_enclosing_cvrect_next_frame);

    p_c_x.create(valid_image_next.size(), CV_32FC1);

    // Fill p_c_x to conclude initialization
    #pragma omp parallel for
    for(auto y = 0; y < valid_image_next.rows; ++y)
    {
        const FLOAT_TYPE* fg_row_ptr = fg_likelihood.ptr<FLOAT_TYPE>(y);
        const FLOAT_TYPE* bg_row_ptr = bg_likelihood.ptr<FLOAT_TYPE>(y);

        FLOAT_TYPE* p_c_x_ptr = p_c_x.ptr<FLOAT_TYPE>(y);

        for (int x=0; x<valid_image_next.cols; ++x)
        {
            const FLOAT_TYPE p_c_x_f = fg_row_ptr[x];
            const FLOAT_TYPE p_c_x_b = bg_row_ptr[x];
            p_c_x_ptr[x] = p_c_x_f / ( p_c_x_b + p_c_x_f + std::numeric_limits<FLOAT_TYPE>::epsilon());
        }
    }

    this->edge_initialized = true;
    return true;
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE SnapcutPairwiseImpl_temporalGMM::GetCost(const cv::Point &coordinates_a,
                                                    const cv::Point &coordinates_b)
// -----------------------------------------------------------------------------------
{
    if(params.block_get_cost)
        return 0.f;

    assert(this->is_initialized);
    assert(this->edge_initialized);

    if(coordinates_a.x < 0 || coordinates_a.y < 0 ||
       coordinates_a.x >= this->matrix_a.cols || coordinates_a.y >= this->matrix_a.rows ||
       coordinates_b.x < 0 || coordinates_b.y < 0 ||
       coordinates_b.x >= this->matrix_a.cols || coordinates_b.y >= this->matrix_a.rows)
       return std::numeric_limits<FLOAT_TYPE>::infinity();

    if(edge_impossible)
    {
        if(std::abs(coordinates_a.x - coordinates_b.x) < 2 && std::abs(coordinates_a.y - coordinates_b.y) < 2)
            return this->params.weight_term; //arbitarily large cost that punishes and b being the same point
        
        return 0.f;
    }

    const RotatedRect rect_1(coordinates_a, coordinates_b, params.region_height, true );
    const RotatedRect rect_2(coordinates_a, coordinates_b, params.region_height, false);

    RotatedRect* rect_foreground;
    RotatedRect* rect_background;

    const FLOAT_TYPE sum_over_rect1 = static_cast<FLOAT_TYPE>(rect_1.SumOver(integral_mask)[0]);
    const FLOAT_TYPE sum_over_rect2 = static_cast<FLOAT_TYPE>(rect_2.SumOver(integral_mask)[0]);

    if(sum_over_rect1 > sum_over_rect2)
    {
        rect_foreground = new RotatedRect(rect_1);
        rect_background = new RotatedRect(rect_2);
    }
    else
    {
        rect_foreground = new RotatedRect(rect_2);
        rect_background = new RotatedRect(rect_1);
    }

    std::vector<cv::Vec3f> colors_fg, colors_bg;
    std::vector<FLOAT_TYPE> distances_fg, distances_bg;
    std::vector<cv::Point> points_fg, points_bg;

    // matrix_a is next_frame
    rect_foreground->BuildDistanceAndColorVectors(this->matrix_a, colors_fg, distances_fg, points_fg, true);
    rect_background->BuildDistanceAndColorVectors(this->matrix_a, colors_bg, distances_bg, points_bg, true);

    delete rect_background;
    delete rect_foreground;

    FLOAT_TYPE normalizer = 0.f;
    FLOAT_TYPE fc = 0;

    // TODO: parallelize with reduction
    for(size_t p = 0; p < points_fg.size(); ++p)
    {
        const int x = std::max(points_fg[p].x - valid_enclosing_cvrect.x, 0);
        const int y = std::max(points_fg[p].y - valid_enclosing_cvrect.y, 0);

        if (x > p_c_x.cols || y > p_c_x.rows)
            continue;

        const FLOAT_TYPE w_c_x = 1.f-std::exp( -distances_fg[p]*distances_fg[p]/(params.sigma_color*params.sigma_color) );

        fc += 1.f - (p_c_x.at<float>(y, x)) * w_c_x;
        normalizer += w_c_x;
    }

    // TODO: parallelize with reduction
    for(size_t p = 0; p < points_bg.size(); ++p)
    {
        const int x = std::max(points_bg[p].x - valid_enclosing_cvrect.x, 0);
        const int y = std::max(points_bg[p].y - valid_enclosing_cvrect.y, 0);

        if (x > p_c_x.cols || y > p_c_x.rows)
            continue;

        const FLOAT_TYPE w_c_x = 1.f - std::exp(-distances_bg[p]*distances_bg[p]/(params.sigma_color*params.sigma_color));

        fc += p_c_x.at<float>(y, x)*w_c_x;
        normalizer += w_c_x;
    }

    // Fc=1-fc is confidence, so I need to return fc only
    return fc/(normalizer + std::numeric_limits<FLOAT_TYPE>::epsilon());
}

// -----------------------------------------------------------------------------------
cv::Ptr<SnapcutPairwise> SnapcutPairwise::createPairwiseTerm(const SnapcutPairwise::Params& parameters)
// -----------------------------------------------------------------------------------
{
    return cv::Ptr<SnapcutPairwiseImpl_temporalGMM>(new SnapcutPairwiseImpl_temporalGMM(parameters));
}

