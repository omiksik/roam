#include "EnergyTerms.h"

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

//-----------------------------------------------------------------------------------------------------------------------
//----------------------------------------------- Common methods of base classes ----------------------------------------
//-----------------------------------------------------------------------------------------------------------------------

using namespace ROAM;

// -----------------------------------------------------------------------------------
UnaryTerm::UnaryTerm() : is_initialized(false)
// -----------------------------------------------------------------------------------
{
}

// -----------------------------------------------------------------------------------
UnaryTerm::~UnaryTerm()
// -----------------------------------------------------------------------------------
{
}

// -----------------------------------------------------------------------------------
bool UnaryTerm::Init(const cv::Mat &image)
// -----------------------------------------------------------------------------------
{
    if(this->is_initialized)
        return false;

    if(image.empty())
        return false;

    const bool impl_initialized = initImpl(image);

    if(impl_initialized)
        this->is_initialized = true;

    return impl_initialized;
}

// -----------------------------------------------------------------------------------
bool UnaryTerm::Update(const cv::Mat &image)
// -----------------------------------------------------------------------------------
{
    if(!this->is_initialized)
        return false;

    return updateImpl(image);
}

// -----------------------------------------------------------------------------------
cv::Size UnaryTerm::GetUnariesDimension() const
// -----------------------------------------------------------------------------------
{
    return this->unaries.size();
}

// -----------------------------------------------------------------------------------
cv::Mat UnaryTerm::GetUnaries() const
// -----------------------------------------------------------------------------------
{
    return this->unaries;
}

// -----------------------------------------------------------------------------------
bool UnaryTerm::IsInitialized() const
// -----------------------------------------------------------------------------------
{
    return this->is_initialized;
}

// -----------------------------------------------------------------------------------
cv::Ptr<UnaryTerm> UnaryTerm::create(const cv::String &unaryType)
// -----------------------------------------------------------------------------------
{
    BOILERPLATE_CODE_UNARY("GRAD", GradientUnary);
    BOILERPLATE_CODE_UNARY("PATCHGRAD", GradientUnary);
    BOILERPLATE_CODE_UNARY("GRADDT", GradientUnary);
    BOILERPLATE_CODE_UNARY("GENE", GradientUnary);
    return cv::Ptr<UnaryTerm>();
}

//-------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
PairwiseTerm::PairwiseTerm() : is_initialized(false)
// -----------------------------------------------------------------------------------
{
}
    
// -----------------------------------------------------------------------------------
PairwiseTerm::~PairwiseTerm()
// -----------------------------------------------------------------------------------
{
}

// -----------------------------------------------------------------------------------
bool PairwiseTerm::Init(const cv::Mat& matrix_a, const cv::Mat& matrix_b )
// -----------------------------------------------------------------------------------
{
    if(this->is_initialized)
        return false;

    const bool impl_initialized = initImpl(matrix_a, matrix_b);

    if(impl_initialized)
        this->is_initialized = true;

    return impl_initialized;
}

// -----------------------------------------------------------------------------------
bool PairwiseTerm::Update(const cv::Mat &matrix_a, const cv::Mat &matrix_b)
// -----------------------------------------------------------------------------------
{
    if(!this->is_initialized)
        return false;

    return updateImpl(matrix_a, matrix_b);
}

// -----------------------------------------------------------------------------------
void PairwiseTerm::GetCosts(const std::vector<cv::Point> &coordinates_a, 
                            const std::vector<cv::Point> &coordinates_b,
                            std::vector<FLOAT_TYPE> &costs)
// -----------------------------------------------------------------------------------
{
    assert(coordinates_a.size() == coordinates_b.size());

    costs.resize(coordinates_a.size());

    #pragma omp parallel for
    for(auto c = 0; c < coordinates_a.size(); ++c)
        costs[c] = GetCost(coordinates_a[c], coordinates_b[c]);
}

// -----------------------------------------------------------------------------------
bool PairwiseTerm::IsInitialized() const
// -----------------------------------------------------------------------------------
{
    return this->is_initialized;
}

// -----------------------------------------------------------------------------------
cv::Ptr<PairwiseTerm> PairwiseTerm::create(const cv::String &pairwiseType)
// -----------------------------------------------------------------------------------
{
    BOILERPLATE_CODE_PAIRWISE("NORM", NormPairwise);
    BOILERPLATE_CODE_PAIRWISE("TEMPNORM", TempNormPairwise);
    return cv::Ptr<PairwiseTerm>();
}

//-----------------------------------------------------------------------------------------------------------------------
//----------------------------------------------- Implementation of GradientUnary ---------------------------------------
//-----------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
class GradientUnaryImpl : public GradientUnary
// -----------------------------------------------------------------------------------
{
public:
    explicit GradientUnaryImpl( const GradientUnary::Params &parameters = GradientUnary::Params() );

    FLOAT_TYPE GetCost(const cv::Point &coordinates) override;

    FLOAT_TYPE GetWeight() const override;

protected:

    bool initImpl( const cv::Mat& /*image*/ ) override;
    bool updateImpl( const cv::Mat& image ) override;

    cv::Mat gradientMagnitude(const cv::Mat &image) const;
};

// -----------------------------------------------------------------------------------
cv::Ptr<GradientUnary> GradientUnary::createUnaryTerm(const GradientUnary::Params& parameters)
// -----------------------------------------------------------------------------------
{
    return cv::Ptr<GradientUnaryImpl>( new GradientUnaryImpl(parameters) );
}

// -----------------------------------------------------------------------------------
GradientUnaryImpl::GradientUnaryImpl(const GradientUnary::Params &parameters)
// -----------------------------------------------------------------------------------
{
    this->is_initialized = false;
    this->unaries = cv::Mat();
    this->params = parameters;
}

// -----------------------------------------------------------------------------------
cv::Mat GradientUnaryImpl::gradientMagnitude(const cv::Mat &image) const
// -----------------------------------------------------------------------------------
{
    cv::Mat mag;
    cv::Mat tmp, gx, gy;

    switch(this->params.grad_type)
    {
        case SOBEL:
        default:
        {
            if(image.channels() != 1)
                cv::cvtColor(image, tmp, cv::COLOR_RGB2GRAY);
            else
                tmp = image;

            cv::GaussianBlur(tmp, tmp, cv::Size(0, 0), this->params.smooth_factor, this->params.smooth_factor);

            cv::Sobel(tmp, gx, CV_32FC1, 1, 0, this->params.kernel_size);
            cv::Sobel(tmp, gy, CV_32FC1, 0, 1, this->params.kernel_size);

            mag = cv::Mat(gx.rows, gx.cols, gx.type());
            cv::magnitude(gx, gy, mag);

            tmp = 1.f-mag;
            cv::normalize(tmp, mag, 0, 1.0, cv::NORM_MINMAX, tmp.depth());

            break;
        }
        case SCHARR:
        {
            if(image.channels() != 1)
                cv::cvtColor(image, tmp, cv::COLOR_RGB2GRAY);
            else
                tmp = image;

            cv::GaussianBlur(tmp, tmp, cv::Size(0, 0), this->params.smooth_factor, this->params.smooth_factor);

            cv::Scharr(tmp, gx, CV_32FC1, 1, 0);
            cv::Scharr(tmp, gy, CV_32FC1, 0, 1);

            mag = cv::Mat(gx.rows, gx.cols, gx.type());
            cv::magnitude(gx, gy, mag);

            tmp = 1.f - mag;
            cv::normalize(tmp, mag, 0, 1.0, cv::NORM_MINMAX, tmp.depth());
            break;
        }
        case LAPLACIAN:
        {
            if(image.channels() != 1)
                cv::cvtColor(image, tmp, cv::COLOR_RGB2GRAY);
            else
                tmp = image;

            cv::GaussianBlur(tmp, tmp, cv::Size(0, 0), this->params.smooth_factor, this->params.smooth_factor);

            cv::Laplacian(tmp, mag, CV_32FC1, this->params.kernel_size);

            tmp = 1.f - mag;
            cv::normalize(tmp, mag, 0, 1.0, cv::NORM_MINMAX, tmp.depth());
            break;
        }
#ifdef HAVE_OPENCV_CONTRIB
        case PIOTR_FORESTS:
        {
            image.convertTo(tmp, cv::DataType<FLOAT_TYPE>::type, 1.f/255.f);
            cv::Ptr<cv::ximgproc::StructuredEdgeDetection> p_d_edge_detector = cv::ximgproc::createStructuredEdgeDetection(params.model_name_trained_detector);
            p_d_edge_detector->detectEdges(tmp, mag);
            mag=1.f-mag;
            break;
        }
#endif
    }

    return mag;
}

// -----------------------------------------------------------------------------------
bool GradientUnaryImpl::initImpl(const cv::Mat &image)
// -----------------------------------------------------------------------------------
{
    this->unaries = gradientMagnitude(image);
    return !unaries.empty();
}

// -----------------------------------------------------------------------------------
bool GradientUnaryImpl::updateImpl(const cv::Mat &image)
// -----------------------------------------------------------------------------------
{
    this->unaries = gradientMagnitude(image);
    return !unaries.empty();
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE GradientUnaryImpl::GetCost(const cv::Point &coordinates)
// -----------------------------------------------------------------------------------
{
    if (coordinates.x < 0 || coordinates.y < 0 ||
        coordinates.x >= this->unaries.cols || coordinates.y >= this->unaries.rows)
        return std::numeric_limits<FLOAT_TYPE>::infinity();

    return this->unaries.at<FLOAT_TYPE>(coordinates); //Keep it in mag better
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE GradientUnaryImpl::GetWeight() const
// -----------------------------------------------------------------------------------
{
    return this->params.weight;
}

// -----------------------------------------------------------------------------------
void GradientUnary::Params::Read(const cv::FileNode& fn)
// -----------------------------------------------------------------------------------
{
    int buffer;
    fn["gradient_type"] >> buffer;
    this->grad_type = static_cast<GradType>(buffer);
    fn["gradient_kernel_size"] >> this->kernel_size;
    fn["gradient_weight"] >> this->weight;
}

// -----------------------------------------------------------------------------------
void GradientUnary::Params::Write(cv::FileStorage& fs) const
// -----------------------------------------------------------------------------------
{
    fs << "gradient_type" << static_cast<int>(this->grad_type);
    fs << "gradient_kernel_size" << this->kernel_size;
    fs << "gradient_weight" << this->weight;
}

//-----------------------------------------------------------------------------------------------------------------------
//----------------------------------------------- Implementation of GradientDTUnary -------------------------------------
//-----------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
class GradientDTUnaryImpl : public GradientDTUnary
// -----------------------------------------------------------------------------------
{
public:
    explicit GradientDTUnaryImpl( const GradientDTUnary::Params &parameters = GradientDTUnary::Params() );

    FLOAT_TYPE GetCost(const cv::Point &coordinates) override;

    FLOAT_TYPE GetWeight() const override;

protected:

    bool initImpl( const cv::Mat& /*image*/ ) override;
    bool updateImpl( const cv::Mat& image ) override;

    cv::Mat gradientMagnitude(const cv::Mat &image) const;
};

// -----------------------------------------------------------------------------------
cv::Ptr<GradientDTUnary> GradientDTUnary::createUnaryTerm(const GradientDTUnary::Params &parameters)
// -----------------------------------------------------------------------------------
{
    return cv::Ptr<GradientDTUnaryImpl>(new GradientDTUnaryImpl(parameters));
}

// -----------------------------------------------------------------------------------
GradientDTUnaryImpl::GradientDTUnaryImpl(const GradientDTUnary::Params &parameters)
// -----------------------------------------------------------------------------------
{
    this->is_initialized = false;
    this->unaries = cv::Mat();
    this->params = parameters;
}

// -----------------------------------------------------------------------------------
cv::Mat GradientDTUnaryImpl::gradientMagnitude(const cv::Mat &image) const
// -----------------------------------------------------------------------------------
{
    cv::Mat mag;
    cv::Mat tmp, gx, gy;

    switch(this->params.grad_type)
    {
        case SOBEL:
        default:
        {
            if(image.channels() != 1)
                cv::cvtColor(image, tmp, cv::COLOR_RGB2GRAY);
            else
                tmp = image;

            cv::GaussianBlur(tmp, tmp, cv::Size(0, 0), this->params.smooth_factor, this->params.smooth_factor);

            cv::Sobel(tmp, gx, CV_32FC1, 1, 0, this->params.kernel_size);
            cv::Sobel(tmp, gy, CV_32FC1, 0, 1, this->params.kernel_size);

            mag = cv::Mat(gx.rows, gx.cols, gx.type());
            cv::magnitude(gx, gy, mag);

            tmp = 1.f-mag;

            break;
        }
        case SCHARR:
        {
            if(image.channels() != 1)
                cv::cvtColor(image, tmp, cv::COLOR_RGB2GRAY);
            else
                tmp = image;

            cv::GaussianBlur(tmp, tmp, cv::Size(0, 0), this->params.smooth_factor, this->params.smooth_factor);

            cv::Scharr(tmp, gx, CV_32FC1, 1, 0);
            cv::Scharr(tmp, gy, CV_32FC1, 0, 1);

            mag = cv::Mat(gx.rows, gx.cols, gx.type());
            cv::magnitude(gx, gy, mag);

            tmp = 1.f - mag;
            break;
        }
        case LAPLACIAN:
        {
            if(image.channels() != 1)
                cv::cvtColor(image, tmp, cv::COLOR_RGB2GRAY);
            else
                tmp = image;

            cv::GaussianBlur(tmp, tmp, cv::Size(0, 0), this->params.smooth_factor, this->params.smooth_factor);

            cv::Laplacian(tmp, mag, CV_32FC1, this->params.kernel_size);

            tmp = 1.f - mag;
            break;
        }
#ifdef HAVE_OPENCV_CONTRIB
        case PIOTR_FORESTS:
        {
            image.convertTo(tmp, cv::DataType<FLOAT_TYPE>::type, 1.f/255.f);
            cv::Ptr<cv::ximgproc::StructuredEdgeDetection> p_d_edge_detector = cv::ximgproc::createStructuredEdgeDetection(params.model_name_trained_detector);
            p_d_edge_detector->detectEdges(tmp, mag);

            tmp = mag.clone();
            break;
        }
#endif
    }

    cv::normalize(tmp, mag, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::Mat dist;
    cv::distanceTransform(mag, dist, CV_DIST_L2, 3);
    tmp = dist.clone();
    cv::normalize(tmp, dist, 0, 1., cv::NORM_MINMAX, CV_32FC1);

    return 1.f - dist;
}

// -----------------------------------------------------------------------------------
bool GradientDTUnaryImpl::initImpl(const cv::Mat &image)
// -----------------------------------------------------------------------------------
{
    this->unaries = gradientMagnitude(image);
    return !unaries.empty();
}

// -----------------------------------------------------------------------------------
bool GradientDTUnaryImpl::updateImpl(const cv::Mat &image)
// -----------------------------------------------------------------------------------
{
    this->unaries = gradientMagnitude(image);

    return !unaries.empty();
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE GradientDTUnaryImpl::GetCost(const cv::Point &coordinates)
// -----------------------------------------------------------------------------------
{
    if (coordinates.x<0 || coordinates.y<0 ||
        coordinates.x>=this->unaries.cols || coordinates.y>=this->unaries.rows)
        return std::numeric_limits<FLOAT_TYPE>::infinity();

    return this->unaries.at<FLOAT_TYPE>(coordinates); //Keep it in mag better
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE GradientDTUnaryImpl::GetWeight() const
// -----------------------------------------------------------------------------------
{
    return this->params.weight;
}

// -----------------------------------------------------------------------------------
void GradientDTUnary::Params::Read(const cv::FileNode& fn)
// -----------------------------------------------------------------------------------
{}

// -----------------------------------------------------------------------------------
void GradientDTUnary::Params::Write(cv::FileStorage& fs) const
// -----------------------------------------------------------------------------------
{}

//-----------------------------------------------------------------------------------------------------------------------
//------------------------------------------------ Implementation of GenericUnary ---------------------------------------
//-----------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
class GenericUnaryImpl : public GenericUnary
// -----------------------------------------------------------------------------------
{
public:
    explicit GenericUnaryImpl( const GenericUnary::Params &parameters = GenericUnary::Params() );

    FLOAT_TYPE GetCost(const cv::Point &coordinates) override;
    FLOAT_TYPE GetWeight() const override;

protected:

    bool initImpl( const cv::Mat& /*image*/ ) override;
    bool updateImpl( const cv::Mat& image ) override;
};

// -----------------------------------------------------------------------------------
cv::Ptr<GenericUnary> GenericUnary::createUnaryTerm(const GenericUnary::Params& parameters)
// -----------------------------------------------------------------------------------
{
    return cv::Ptr<GenericUnaryImpl>( new GenericUnaryImpl(parameters) );
}

// -----------------------------------------------------------------------------------
GenericUnaryImpl::GenericUnaryImpl(const GenericUnary::Params &parameters)
// -----------------------------------------------------------------------------------
{
    this->is_initialized = false;
    this->unaries = cv::Mat();
    this->params = parameters;
}

// -----------------------------------------------------------------------------------
bool GenericUnaryImpl::initImpl(const cv::Mat &map)
// -----------------------------------------------------------------------------------
{
    this->unaries = map.clone();
    return !unaries.empty();
}

// -----------------------------------------------------------------------------------
bool GenericUnaryImpl::updateImpl(const cv::Mat &map)
// -----------------------------------------------------------------------------------
{
    this->unaries = map.clone();
    return !unaries.empty();
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE GenericUnaryImpl::GetCost(const cv::Point &coordinates)
// -----------------------------------------------------------------------------------
{
    if(coordinates.x < 0 || coordinates.y < 0 ||
        coordinates.x >= this->unaries.cols || coordinates.y >= this->unaries.rows)
        return std::numeric_limits<FLOAT_TYPE>::infinity();

    switch(this->params.pooling_type)
    {
        default:
        case PoolingType::NO_POOL:
            return this->unaries.at<FLOAT_TYPE>(coordinates);
        case PoolingType::MEAN_POOL:
        {
            cv::Rect sub_rect(static_cast<int>(coordinates.x - this->params.pooling_win_size / 2.f), 
                              static_cast<int>(coordinates.y - this->params.pooling_win_size / 2.f),
                              static_cast<int>(this->params.pooling_win_size),
                              static_cast<int>(this->params.pooling_win_size));
            sub_rect &= cv::Rect(0, 0, this->unaries.cols, this->unaries.rows);
            const cv::Mat sub_win = this->unaries(sub_rect);
            return static_cast<FLOAT_TYPE>(cv::mean(sub_win)[0]);
        }
        case PoolingType::MAX_POOL:
        {
            cv::Rect sub_rect(static_cast<int>(coordinates.x - this->params.pooling_win_size / 2.f), 
                              static_cast<int>(coordinates.y - this->params.pooling_win_size / 2.f),
                              static_cast<int>(this->params.pooling_win_size), 
                              static_cast<int>(this->params.pooling_win_size));
            sub_rect &= cv::Rect(0, 0, this->unaries.cols, this->unaries.rows);
            const cv::Mat sub_win = this->unaries(sub_rect);
            double minx, maxx;
            cv::minMaxIdx(sub_win, &minx, &maxx);
            return static_cast<FLOAT_TYPE>(maxx);
        }
    }
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE GenericUnaryImpl::GetWeight() const
// -----------------------------------------------------------------------------------
{
    return this->params.weight;
}

//-----------------------------------------------------------------------------------------------------------------------
//----------------------------------------------- Implementation of DistanceUnary ---------------------------------------
//-----------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
void DistanceUnary::SetPastDistance(const FLOAT_TYPE dist)
// -----------------------------------------------------------------------------------
{
    this->past_distance = dist;
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE DistanceUnary::GetPastDistance() const
// -----------------------------------------------------------------------------------
{
    return this->past_distance;
}

// -----------------------------------------------------------------------------------
class DistanceUnaryImpl : public DistanceUnary
// -----------------------------------------------------------------------------------
{
public:
    explicit DistanceUnaryImpl( const DistanceUnary::Params &parameters = DistanceUnary::Params() );

    FLOAT_TYPE GetCost(const cv::Point &coordinates) override;
    FLOAT_TYPE GetWeight() const override;

protected:

    bool initImpl( const cv::Mat& /*image*/ ) override;
    bool updateImpl( const cv::Mat& image ) override;
};

// -----------------------------------------------------------------------------------
cv::Ptr<DistanceUnary> DistanceUnary::createUnaryTerm(const DistanceUnary::Params& parameters)
// -----------------------------------------------------------------------------------
{
    return cv::Ptr<DistanceUnaryImpl>( new DistanceUnaryImpl(parameters) );
}

// -----------------------------------------------------------------------------------
DistanceUnaryImpl::DistanceUnaryImpl(const DistanceUnary::Params &parameters)
// -----------------------------------------------------------------------------------
{
    this->past_distance = 0.f;
    this->is_initialized = false;
    this->unaries = cv::Mat();
    this->params = parameters;
}

// -----------------------------------------------------------------------------------
bool DistanceUnaryImpl::initImpl(const cv::Mat &map)
// -----------------------------------------------------------------------------------
{
    this->unaries = map.clone();
    return !unaries.empty();
}

// -----------------------------------------------------------------------------------
bool DistanceUnaryImpl::updateImpl(const cv::Mat &map)
// -----------------------------------------------------------------------------------
{
    this->unaries = map.clone();
    return !unaries.empty();
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE DistanceUnaryImpl::GetCost(const cv::Point &coordinates)
// -----------------------------------------------------------------------------------
{
    FLOAT_TYPE accum_distance = 0.f;

    if(this->unaries.rows == 0)
        return 0.f;

    if(this->past_distance >= 0.f)
    {
        // TODO parallelize with reduction
        for(int row = 0; row<this->unaries.rows; ++row)
            accum_distance += static_cast<FLOAT_TYPE>(cv::norm(coordinates - this->unaries.at<cv::Point>(row)));

        return std::abs(accum_distance / (static_cast<FLOAT_TYPE>(this->unaries.rows) + std::numeric_limits<FLOAT_TYPE>::epsilon()) - this->past_distance );
    }
    
    return 0.f;
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE DistanceUnaryImpl::GetWeight() const
// -----------------------------------------------------------------------------------
{
    return this->params.weight;
}


//-----------------------------------------------------------------------------------------------------------------------
//----------------------------------------------- Implementation of NormPairwise ----------------------------------------
//-----------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
class NormPairwiseImpl : public NormPairwise
// -----------------------------------------------------------------------------------
{
public:
    explicit NormPairwiseImpl(const NormPairwise::Params &parameters = NormPairwise::Params());

    FLOAT_TYPE GetCost(const cv::Point &coordinates_a, const cv::Point &coordinates_b) override;

    FLOAT_TYPE GetWeight() const override;

protected:

    bool initImpl(const cv::Mat& mat_a, const cv::Mat& mat_b) override;
    bool updateImpl(const cv::Mat& mat_a, const cv::Mat& mat_b) override;
};

// -----------------------------------------------------------------------------------
cv::Ptr<NormPairwise> NormPairwise::createPairwiseTerm(const NormPairwise::Params& parameters)
// -----------------------------------------------------------------------------------
{
    return cv::Ptr<NormPairwiseImpl>(new NormPairwiseImpl(parameters));
}

// -----------------------------------------------------------------------------------
NormPairwiseImpl::NormPairwiseImpl(const NormPairwise::Params &parameters)
// -----------------------------------------------------------------------------------
{
    this->is_initialized = false;
    this->matrix_a = cv::Mat();
    this->matrix_b = cv::Mat();
    this->params = parameters;
}

// -----------------------------------------------------------------------------------
bool NormPairwiseImpl::initImpl(const cv::Mat&, const cv::Mat&)
// -----------------------------------------------------------------------------------
{
    return true;
}

// -----------------------------------------------------------------------------------
bool NormPairwiseImpl::updateImpl(const cv::Mat&, const cv::Mat&)
// -----------------------------------------------------------------------------------
{
    return true;
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE NormPairwiseImpl::GetWeight() const
// -----------------------------------------------------------------------------------
{
    return this->params.weight;
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE NormPairwiseImpl::GetCost(const cv::Point &coordinates_a, const cv::Point &coordinates_b)
// -----------------------------------------------------------------------------------
{
    switch(this->params.norm_type)
    {
        case NormPairwise::L2_SQR:
            return static_cast<FLOAT_TYPE>((coordinates_a.x-coordinates_b.x)*(coordinates_a.x-coordinates_b.x) +
                                           (coordinates_a.y-coordinates_b.y)*(coordinates_a.y-coordinates_b.y));
        case NormPairwise::L2:
            return static_cast<FLOAT_TYPE>(cv::norm(coordinates_a-coordinates_b));
        case NormPairwise::L1:
            return static_cast<FLOAT_TYPE>(std::abs(coordinates_a.x-coordinates_b.x) + std::abs(coordinates_a.y-coordinates_b.y));
        case NormPairwise::LINF:
            return static_cast<FLOAT_TYPE>(std::max(std::abs(coordinates_a.x - coordinates_b.x), std::abs(coordinates_a.y - coordinates_b.y)));
        default:
            throw std::invalid_argument("NormPairwiseImpl::getCost - unknown norm: " + std::to_string(this->params.norm_type));
    }
}

// -----------------------------------------------------------------------------------
void NormPairwise::Params::Read(const cv::FileNode &fn)
// -----------------------------------------------------------------------------------
{
    int buffer;
    fn["norm_type"] >> buffer;
    this->norm_type = static_cast<NormType>(buffer);
    fn["norm_weight"] >> this->weight;
}

// -----------------------------------------------------------------------------------
void NormPairwise::Params::Write(cv::FileStorage &fs) const
// -----------------------------------------------------------------------------------
{
    fs << "norm_type" << static_cast<int>(this->norm_type);
    fs << "norm_weight" << this->weight;
}

//-----------------------------------------------------------------------------------------------------------------------
//--------------------------------------------- Implementation of TempNormPairwise --------------------------------------
//-----------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
class TempNormPairwiseImpl : public TempNormPairwise
// -----------------------------------------------------------------------------------
{
public:
    explicit TempNormPairwiseImpl(const TempNormPairwise::Params &parameters = TempNormPairwise::Params());

    FLOAT_TYPE GetCost(const cv::Point &coordinates_a, const cv::Point &coordinates_b) override;

    FLOAT_TYPE GetWeight() const override;

protected:

    bool initImpl(const cv::Mat& mat_a, const cv::Mat& mat_b) override;
    bool updateImpl(const cv::Mat& mat_a, const cv::Mat& mat_b) override;
};

// -----------------------------------------------------------------------------------
cv::Ptr<TempNormPairwise> TempNormPairwise::createPairwiseTerm(const TempNormPairwise::Params& parameters)
// -----------------------------------------------------------------------------------
{
    return cv::Ptr<TempNormPairwiseImpl>(new TempNormPairwiseImpl(parameters));
}

// -----------------------------------------------------------------------------------
TempNormPairwiseImpl::TempNormPairwiseImpl(const TempNormPairwise::Params &parameters)
// -----------------------------------------------------------------------------------
{
    this->is_initialized = false;
    this->matrix_a = cv::Mat();
    this->matrix_b = cv::Mat();
    this->params = parameters;
}

// -----------------------------------------------------------------------------------
bool TempNormPairwiseImpl::initImpl(const cv::Mat &m_a, const cv::Mat&)
// -----------------------------------------------------------------------------------
{
    this->matrix_a = m_a.clone();
    return !this->matrix_a.empty();
}

// -----------------------------------------------------------------------------------
bool TempNormPairwiseImpl::updateImpl(const cv::Mat &m_a, const cv::Mat&)
// -----------------------------------------------------------------------------------
{
    this->matrix_a = m_a.clone();
    return !this->matrix_a.empty();
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE TempNormPairwiseImpl::GetWeight() const
// -----------------------------------------------------------------------------------
{
    return this->params.weight;
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE TempNormPairwiseImpl::GetCost(const cv::Point &coordinates_a, const cv::Point &coordinates_b)
// -----------------------------------------------------------------------------------
{
    if(this->matrix_a.at<FLOAT_TYPE>(0) < 0)
        return 0;

    const FLOAT_TYPE curr_diff = static_cast<FLOAT_TYPE>(cv::norm(coordinates_a - coordinates_b));
    return std::abs(curr_diff - this->matrix_a.at<FLOAT_TYPE>(0));
}

//-----------------------------------------------------------------------------------------------------------------------
//--------------------------------------------- Implementation of GenericPairwise --------------------------------------
//-----------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
class TempAnglePairwiseImpl : public TempAnglePairwise
// -----------------------------------------------------------------------------------
{
public:
    explicit TempAnglePairwiseImpl(const TempAnglePairwise::Params &parameters = TempAnglePairwise::Params());

    FLOAT_TYPE GetCost(const cv::Point &coordinates_a, const cv::Point &coordinates_b) override;
    FLOAT_TYPE GetWeight() const override;

protected:
    bool initImpl(const cv::Mat& mat_a, const cv::Mat& mat_b) override;
    bool updateImpl(const cv::Mat& mat_a, const cv::Mat& mat_b) override;
};

// -----------------------------------------------------------------------------------
cv::Ptr<TempAnglePairwise> TempAnglePairwise::createPairwiseTerm(const TempAnglePairwise::Params& parameters)
// -----------------------------------------------------------------------------------
{
    return cv::Ptr<TempAnglePairwiseImpl>(new TempAnglePairwiseImpl(parameters));
}

// -----------------------------------------------------------------------------------
TempAnglePairwiseImpl::TempAnglePairwiseImpl(const TempAnglePairwise::Params &parameters)
// -----------------------------------------------------------------------------------
{
    this->is_initialized = false;
    this->matrix_a = cv::Mat();
    this->matrix_b = cv::Mat();
    this->params = parameters;
}

// -----------------------------------------------------------------------------------
bool TempAnglePairwiseImpl::initImpl(const cv::Mat &m_a, const cv::Mat&)
// -----------------------------------------------------------------------------------
{
    this->matrix_a = m_a.clone();
    return !this->matrix_a.empty();
}

// -----------------------------------------------------------------------------------
bool TempAnglePairwiseImpl::updateImpl(const cv::Mat &m_a, const cv::Mat &m_b)
// -----------------------------------------------------------------------------------
{
    this->matrix_a = m_a.clone();
    return !this->matrix_a.empty();
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE TempAnglePairwiseImpl::GetWeight() const
// -----------------------------------------------------------------------------------
{
    return this->params.weight;
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE TempAnglePairwiseImpl::GetCost(const cv::Point &coordinates_a, const cv::Point &coordinates_b)
// -----------------------------------------------------------------------------------
{
    if (this->matrix_a.at<FLOAT_TYPE>(0)<=-5*CV_PI)
        return 0;

    const FLOAT_TYPE curr_angle = static_cast<FLOAT_TYPE>(std::atan2(coordinates_b.y-coordinates_a.y, coordinates_b.x-coordinates_a.x));
    return std::abs(curr_angle - this->matrix_a.at<FLOAT_TYPE>(0));
}

//-----------------------------------------------------------------------------------------------------------------------
//--------------------------------------------- Implementation of GenericPairwise ---------------------------------------
//-----------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
class GenericPairwiseImpl : public GenericPairwise
// -----------------------------------------------------------------------------------
{
public:
    explicit GenericPairwiseImpl(const GenericPairwise::Params &parameters = GenericPairwise::Params());

    FLOAT_TYPE GetCost(const cv::Point &coordinates_a, const cv::Point &coordinates_b) override;
    FLOAT_TYPE GetWeight() const override;

protected:
    bool initImpl(const cv::Mat& mat_a, const cv::Mat& mat_b) override;
    bool updateImpl(const cv::Mat& mat_a, const cv::Mat& mat_b) override;
};

// -----------------------------------------------------------------------------------
cv::Ptr<GenericPairwise> GenericPairwise::createPairwiseTerm(const GenericPairwise::Params& parameters)
// -----------------------------------------------------------------------------------
{
    return cv::Ptr<GenericPairwiseImpl>(new GenericPairwiseImpl(parameters));
}

// -----------------------------------------------------------------------------------
GenericPairwiseImpl::GenericPairwiseImpl(const GenericPairwise::Params &parameters)
// -----------------------------------------------------------------------------------
{
    this->is_initialized = false;
    this->matrix_a = cv::Mat();
    this->matrix_b = cv::Mat();
    this->params = parameters;
}

// -----------------------------------------------------------------------------------
bool GenericPairwiseImpl::initImpl(const cv::Mat &m_a, const cv::Mat&)
// -----------------------------------------------------------------------------------
{
    this->matrix_a = m_a.clone();
    return !this->matrix_a.empty();
}

// -----------------------------------------------------------------------------------
bool GenericPairwiseImpl::updateImpl(const cv::Mat &m_a, const cv::Mat &m_b)
// -----------------------------------------------------------------------------------
{
    this->matrix_a = m_a.clone();
    return !this->matrix_a.empty();
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE GenericPairwiseImpl::GetWeight() const
// -----------------------------------------------------------------------------------
{
    return this->params.weight;
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE GenericPairwiseImpl::GetCost(const cv::Point &coordinates_a, const cv::Point &coordinates_b)
// -----------------------------------------------------------------------------------
{
    const cv::Rect im_rect(cv::Point(), this->matrix_a.size());

    if(!im_rect.contains(coordinates_a) || !im_rect.contains(coordinates_b))
        return std::numeric_limits<FLOAT_TYPE>::infinity();

    // from 8-bit 3-channel image to the buffer
    cv::LineIterator it(this->matrix_a, coordinates_a, coordinates_b, 8);

    // TODO: parallelize with reduction
    FLOAT_TYPE sum = 0.f;
    for(int i = 0; i < it.count; i++, ++it)
        sum += this->matrix_a.at<FLOAT_TYPE>(it.pos());

    sum /= static_cast<FLOAT_TYPE>(it.count) + std::numeric_limits<FLOAT_TYPE>::epsilon();

    return sum;
}
