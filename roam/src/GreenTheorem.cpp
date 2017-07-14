#include "GreenTheorem.h"


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
const cv::Mat& GreenTheoremPairwise::GetMatrixA() const
// -----------------------------------------------------------------------------------
{
    return matrix_a;
}

// -----------------------------------------------------------------------------------
const cv::Mat& GreenTheoremPairwise::GetMatrixB() const
// -----------------------------------------------------------------------------------
{
    return matrix_b;
}

//-----------------------------------------------------------------------------------------------------------------------
//------------------------------------------ Implementation of GreenTheoremPairwise -------------------------------------
//-----------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
class GreenTheoremPairwiseImpl : public GreenTheoremPairwise
// -----------------------------------------------------------------------------------
{
public:
    explicit GreenTheoremPairwiseImpl(const GreenTheoremPairwise::Params &parameters = GreenTheoremPairwise::Params());

    FLOAT_TYPE GetCost(const cv::Point &coordinates_a, const cv::Point &coordinates_b) override;
    FLOAT_TYPE GetWeight() const override;

    bool InitializeEdge() override;

protected:

    bool initImpl(const cv::Mat& color_image, const cv::Mat& integral_neg_fb_ratio) override;
    bool updateImpl(const cv::Mat& mat_a, const cv::Mat& mat_b) override;
};

// -----------------------------------------------------------------------------------
GreenTheoremPairwiseImpl::GreenTheoremPairwiseImpl(const GreenTheoremPairwise::Params &parameters)
// -----------------------------------------------------------------------------------
{
    this->params = parameters;
    this->is_initialized = false;
    this->matrix_a = cv::Mat();
    this->matrix_b = cv::Mat();
}

// -----------------------------------------------------------------------------------
bool GreenTheoremPairwiseImpl::initImpl(const cv::Mat &color_image,
                                        const cv::Mat &integral_neg_fb_ratio)
// -----------------------------------------------------------------------------------
{
    if(color_image.size() == integral_neg_fb_ratio.size())
    {
        this->matrix_a = color_image;
        this->matrix_b = integral_neg_fb_ratio;

        return (!this->matrix_a.empty() && !this->matrix_b.empty());
    }

    return false;
}

// -----------------------------------------------------------------------------------
bool GreenTheoremPairwiseImpl::updateImpl(const cv::Mat& color_image,
                                          const cv::Mat& integral_neg_fb_ratio)
// -----------------------------------------------------------------------------------
{
    if(color_image.size() == integral_neg_fb_ratio.size())
    {
        this->matrix_a = color_image;
        this->matrix_b = integral_neg_fb_ratio;

        return (!this->matrix_a.empty() && !this->matrix_b.empty());
    }

    return false;
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE GreenTheoremPairwiseImpl::GetWeight() const
// -----------------------------------------------------------------------------------
{
    return this->params.weight_term;
}

// -----------------------------------------------------------------------------------
bool GreenTheoremPairwiseImpl::InitializeEdge()
// -----------------------------------------------------------------------------------
{
    return true;
}

// -----------------------------------------------------------------------------------
inline bool plusSign(const cv::Point &a, const cv::Point &b,
                     const size_t id_a, const size_t id_b,
                     const bool counterclockwise = true)
// -----------------------------------------------------------------------------------
{
    assert(a.x != b.x);

    // sanity check (make sure the points are ordered correctly)
    cv::Point2f prev, current;
    if((id_a < id_b) || (id_b == 0 && id_a != 1)) // second condition handles [last - zero] node
    {
        prev = a;
        current = b;
    }
    else
    {
        prev = b;
        current = a;
    }

    if(counterclockwise)
    {
        if(current.x < prev.x)
            return false;
        else if(current.x > prev.x)
            return true;
    }
    else
    {
        if(current.x > prev.x)
            return false;
        else if(current.x < prev.x)
            return true;
    }
    
    throw std::logic_error("GreenTheoremPairwiseImpl::plusSign - this should never happen," 
                           " we've hit some strange combination (this exception was added later on,"
                           " so if things brake now, double check this method!!!)");
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE GreenTheoremPairwiseImpl::GetCost(const cv::Point &coordinates_a,
                                             const cv::Point &coordinates_b)
// -----------------------------------------------------------------------------------
{   
    FLOAT_TYPE cost = 0.0;
    if(this->is_initialized)
    {
        // handle same node separately
        if(coordinates_a == coordinates_b)
            return cost;

        // for all non-vertical edges
        if(coordinates_a.x != coordinates_b.x)
        {
            // ----------------------------------------------
            // determine plus/minus
            const bool plus = plusSign(coordinates_a, coordinates_b,
                                       this->params.id_a, this->params.id_b, this->params.contour_is_ccw);

            // ----------------------------------------------
            // compute all pixels between endpoints
            cv::LineIterator it(this->matrix_a/*???*/, coordinates_a, coordinates_b, 8);

            // ----------------------------------------------
            // define default values
            const size_t default_y = plus ? 0 : std::max(coordinates_b.y, coordinates_a.y) + 10; // 10 some dummy constant
            const size_t n_points = std::abs(coordinates_a.x - coordinates_b.x) + 1;

            std::vector<cv::Point2f> points(n_points, cv::Point2f(0, static_cast<float>(default_y)));

            // ----------------------------------------------
            // find the min/max box per column
            // TODO: can run in parallel with reduction
            for (size_t i = 0; i < it.count; ++i, ++it)
            {
                const cv::Point2f pt = it.pos();
                const size_t pos = static_cast<size_t>(pt.x - std::min(coordinates_b.x, coordinates_a.x));

                if(plus && points[pos].y < pt.y)
                    points[pos] = pt;
                else if(!plus && points[pos].y > pt.y)
                    points[pos] = pt;
            }

            // ----------------------------------------------
            // compute cost
            // TODO: can run in parallel with reduction
            for (int i = 0; i < n_points; i++)
            {
                // split cost for the first and last points
                const FLOAT_TYPE multiplier = (i == 0 || i == (points.size() - 1)) ? 0.5f : 1.0f;

                // add costs
                if(plus)
                    cost += multiplier * this->matrix_b.at<float>(points[i]);
                else
                    cost -= multiplier * this->matrix_b.at<float>(cv::Point(static_cast<int>(points[i].x), static_cast<int>(points[i].y - 1)));
            }
        }
        else // treat vertical edges separately
        {
            if(coordinates_a.y < coordinates_b.y) // top -> down
            {
                cost -= 0.5f * this->matrix_b.at<float>(cv::Point(coordinates_a.x, coordinates_a.y - 1));
                cost += 0.5f * this->matrix_b.at<float>(coordinates_b);
            }
            else // bottom -> up
            {
                cost += 0.5f * this->matrix_b.at<float>(coordinates_a);
                cost -= 0.5f * this->matrix_b.at<float>(cv::Point(coordinates_b.x, coordinates_b.y - 1));
            }
        }
    }
    return cost;
}

// -----------------------------------------------------------------------------------
cv::Ptr<GreenTheoremPairwise> GreenTheoremPairwise::createPairwiseTerm(const GreenTheoremPairwise::Params& parameters)
// -----------------------------------------------------------------------------------
{
    return cv::Ptr<GreenTheoremPairwiseImpl>(new GreenTheoremPairwiseImpl(parameters));
}

