#include "GC_Energy.h"



using namespace ROAM;

// ---------------------------------------------------------------------------
GC_Energy::GC_Energy(const GC_Energy::Params &parameters)
// ---------------------------------------------------------------------------
{
    this->params = parameters;
    model = GlobalModel(GlobalModel::Params(this->params.iterations, this->params.betha_nd, this->params.gamma_gc, this->params.lambda_gc));
    this->is_initialized = false;
}

// ---------------------------------------------------------------------------
bool GC_Energy::Initialized() const
// ---------------------------------------------------------------------------
{
    return is_initialized;
}

// ---------------------------------------------------------------------------
bool GC_Energy::initialize(const cv::Mat &data, const cv::Mat &labels)
// ---------------------------------------------------------------------------
{
    return this->model.Initialize(data, labels);
}

// ---------------------------------------------------------------------------
bool GC_Energy::update(const cv::Mat &data, const cv::Mat &labels)
// ---------------------------------------------------------------------------
{
    return this->model.Update(data, labels);
}

// TODO: why raw pointer and undeclared in header?
// ---------------------------------------------------------------------------
static void
construct8Graph(const cv::Mat& img, const cv::Mat& mask,
                const GlobalModel& model, double lambda,
                const cv::Mat& leftW, const cv::Mat& upleftW, const cv::Mat& upW, const cv::Mat& uprightW,
                std::shared_ptr<GraphType> graph, const cv::Mat &precomputed_contour_likelihood = cv::Mat())
// ---------------------------------------------------------------------------
{
    cv::Mat buffer_er = cv::Mat::zeros( img.size(), CV_32FC1 );

    for(auto y = 0; y < img.rows; ++y)
        for(auto x = 0; x < img.cols; ++x)
        {
            // add node
            const int vtxIdx = graph->add_node();
            const cv::Vec3b &color = img.at<cv::Vec3b>(y, x);

            cv::Vec2f pre_likelihood(0,0);
            if(!precomputed_contour_likelihood.empty())
                pre_likelihood = precomputed_contour_likelihood.at<cv::Vec2f>(y, x);

            // set t-weights
            float fromSource, toSink;
            if (mask.at<uchar>(y, x) == GC_Energy::GCClasses::GC_PR_BGD || mask.at<uchar>(y, x) == GC_Energy::GCClasses::GC_PR_FGD)
            {
                const cv::Vec2f lik_vec = model.ComputeLikelihood(color) + pre_likelihood;
                fromSource = -std::log( lik_vec[1] );
                toSink = -std::log( lik_vec[0] );

                buffer_er.at<float>(y, x) = static_cast<float>(toSink) / (fromSource + std::numeric_limits<float>::epsilon());
            }
            else if(mask.at<uchar>(y, x) == GC_Energy::GCClasses::GC_BGD)
            {
                fromSource = 0;
                toSink = static_cast<float>(lambda);

                buffer_er.at<float>(y, x) = 100;
            }
            else // GC_FGD
            {
                fromSource = static_cast<float>(lambda);
                toSink = 0;

                buffer_er.at<float>(y, x) = 0;
            }
            graph->add_tweights( vtxIdx, fromSource, toSink );

            // set n-weights
            if(x > 0)
            {
                const float &w = static_cast<float>(leftW.at<double>(y, x));
                graph->add_edge( vtxIdx, vtxIdx-1, w, w );
            }
            if(x > 0 && y > 0)
            {
                const float &w = static_cast<float>(upleftW.at<double>(y, x));
                graph->add_edge( vtxIdx, vtxIdx-img.cols-1, w, w );
            }
            if(y > 0)
            {
                const float &w = static_cast<float>(upW.at<double>(y, x));
                graph->add_edge( vtxIdx, vtxIdx-img.cols, w, w );
            }
            if(x < img.cols - 1 && y > 0)
            {
                const float &w = static_cast<float>(uprightW.at<double>(y, x));
                graph->add_edge( vtxIdx, vtxIdx-img.cols+1, w, w );
            }
        }
}


// TODO: why raw pointer and undeclared in header?
// ---------------------------------------------------------------------------
static void estimateSegmentation(std::shared_ptr<GraphType> graph, cv::Mat &mask)
// ---------------------------------------------------------------------------
{
    graph->maxflow();
    
    #pragma omp parallel for
    for(auto y = 0; y < mask.rows; ++y)
        for(auto x = 0; x < mask.cols; ++x)
        {
            const auto &val = mask.at<uchar>(y, x);
            if(val == GC_Energy::GCClasses::GC_PR_BGD || val == GC_Energy::GCClasses::GC_PR_FGD)
            {
                if (graph->what_segment(y * mask.cols + x /*vertex index*/) == GraphType::SOURCE)
                    mask.at<uchar>(y, x) = cv::GC_PR_FGD;
                else
                    mask.at<uchar>(y, x) = cv::GC_PR_BGD;
            }
        }
}

// TODO: why undeclared in header?
// ---------------------------------------------------------------------------
static double calcBeta(const cv::Mat &img)
// ---------------------------------------------------------------------------
{
    double beta = 0;

    #pragma omp parallel for reduction(+:beta)
    for(auto y = 0; y < img.rows; ++y)
        for(auto x = 0; x < img.cols; ++x)
        {
            const cv::Vec3d &color = img.at<cv::Vec3b>(y,x);
            if(x > 0) // left
            {
                const cv::Vec3d diff = color - static_cast<cv::Vec3d>(img.at<cv::Vec3b>(y, x - 1));
                beta += diff.dot(diff);
            }
            if(y > 0 && x > 0) // upleft
            {
                const cv::Vec3d diff = color - static_cast<cv::Vec3d>(img.at<cv::Vec3b>(y - 1, x - 1));
                beta += diff.dot(diff);
            }
            if(y > 0) // up
            {
                const cv::Vec3d diff = color - static_cast<cv::Vec3d>(img.at<cv::Vec3b>(y - 1, x));
                beta += diff.dot(diff);
            }
            if(y > 0 && x < img.cols - 1) // upright
            {
                const cv::Vec3d diff = color - static_cast<cv::Vec3d>(img.at<cv::Vec3b>(y - 1, x + 1));
                beta += diff.dot(diff);
            }
        }

    if(beta <= std::numeric_limits<double>::epsilon())
        beta = 0;
    else
        beta = 1.f / (2 * beta/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2) );

    return beta;
}

// TODO: why undeclared in header?
// ---------------------------------------------------------------------------
static void
calcNWeights(const cv::Mat& img, cv::Mat& leftW, cv::Mat& upleftW, cv::Mat& upW, cv::Mat& uprightW,
             const double beta, const double gamma)
// ---------------------------------------------------------------------------
{
    const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
    leftW.create( img.rows, img.cols, CV_64FC1 );
    upleftW.create( img.rows, img.cols, CV_64FC1 );
    upW.create( img.rows, img.cols, CV_64FC1 );
    uprightW.create( img.rows, img.cols, CV_64FC1 );

    #pragma omp parallel for
    for(auto y = 0; y < img.rows; ++y)
        for(auto x = 0; x < img.cols; ++x)
        {
            const cv::Vec3d &color = img.at<cv::Vec3b>(y, x);
            if(x - 1 >= 0) // left
            {
                const cv::Vec3d diff = color - static_cast<cv::Vec3d>(img.at<cv::Vec3b>(y, x - 1));
                leftW.at<double>(y,x) = gamma * std::exp(-beta*diff.dot(diff));
            }
            else
                leftW.at<double>(y,x) = 0;
            if(x - 1 >= 0 && y - 1 >= 0) // upleft
            {
                const cv::Vec3d diff = color - static_cast<cv::Vec3d>(img.at<cv::Vec3b>(y - 1, x - 1));
                upleftW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                upleftW.at<double>(y,x) = 0;
            if(y-1 >= 0) // up
            {
                const cv::Vec3d diff = color - static_cast<cv::Vec3d>(img.at<cv::Vec3b>(y - 1, x));
                upW.at<double>(y,x) = gamma * std::exp(-beta*diff.dot(diff));
            }
            else
                upW.at<double>(y,x) = 0;
            if(x+1 < img.cols && y-1 >= 0) // upright
            {
                const cv::Vec3d diff = color - static_cast<cv::Vec3d>(img.at<cv::Vec3b>(y - 1, x + 1));
                uprightW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                uprightW.at<double>(y,x) = 0;
        }
}

// ---------------------------------------------------------------------------
static cv::Mat MaskFromContour(const std::vector<cv::Point> &closed_contour, const cv::Size &sz)
// ---------------------------------------------------------------------------
{
    cv::Mat mask(sz, CV_8UC1, GC_Energy::GCClasses::GC_BGD);
    cv::Rect im_rect(cv::Point(), sz);
    cv::Rect disputed_space = cv::minAreaRect(closed_contour).boundingRect();
    const int margin = 10;
    disputed_space.x -= margin;
    disputed_space.y -= margin;
    disputed_space.width += 2*margin;
    disputed_space.height += 2*margin;
    disputed_space &= im_rect;

    mask(disputed_space) = cv::Scalar(GC_Energy::GCClasses::GC_PR_FGD);

    return mask;
}

// ---------------------------------------------------------------------------
static cv::Mat LabelsFromMask(const cv::Mat &mask)
// ---------------------------------------------------------------------------
{
    cv::Mat labels(mask.size(), CV_8UC1, cv::Scalar(0));

    #pragma omp parallel for
    for(auto y = 0; y < mask.rows; ++y)
        for(auto x = 0; x < mask.cols; ++x)
        {
            const auto &val = mask.at<uchar>(y, x);
            if (val == GC_Energy::GCClasses::GC_PR_FGD || val == GC_Energy::GCClasses::GC_FGD)
                labels.at<uchar>(y,x) = 255;
        }

    return labels;
}

// ---------------------------------------------------------------------------
cv::Mat GC_Energy::Segment(const cv::Mat &image_, const cv::Mat &labels_,
                                      const std::vector<cv::Point> &closed_contour,
                                      const cv::Mat &precomputed_contour_likelihood_,
                                      const GlobalModel &global_model, const cv::Rect& slack)
// ---------------------------------------------------------------------------
{
    cv::Mat image, labels, precomputed_contour_likelihood;

    if (slack==cv::Rect())
    {
        image = image_;
        labels = labels_;
        precomputed_contour_likelihood = precomputed_contour_likelihood_;
    }
    else
    {
        image = image_(slack);
        labels = labels_(slack);
        precomputed_contour_likelihood = precomputed_contour_likelihood_(slack);
    }


    if (global_model.initialized)
        this->model = global_model;

    const cv::Mat &img = image;
    cv::Mat mask = labels.clone();


    if( params.iterations <= 0)
        return cv::Mat();

    const double gamma = this->params.gamma_gc;
    const double lambda = this->params.lambda_gc;
    const double beta = calcBeta( img );

    cv::Mat leftW, upleftW, upW, uprightW;
    calcNWeights( img, leftW, upleftW, upW, uprightW, beta, gamma );


    cv::Mat mask_labels( mask.size(), CV_8UC1, cv::Scalar(cv::GC_PR_BGD) );
    cv::Mat tr_mask = (mask/255)*cv::GC_PR_FGD;
    tr_mask.copyTo(mask_labels, mask);

    for(unsigned int i = 0; i < this->params.iterations; ++i )
    {
        int vtxCount = img.cols*img.rows,
            edgeCount = 2*(4*img.cols*img.rows - 3*(img.cols + img.rows) + 2);
        //GraphType *graph = new GraphType(vtxCount, edgeCount);
        std::shared_ptr<GraphType> graph = std::make_shared<GraphType>(vtxCount, edgeCount);

        if (!is_initialized && !model.initialized)
            this->initialize( img, LabelsFromMask(mask_labels) );
        else if(i>0)
            this->update( img, LabelsFromMask(mask_labels) );


        construct8Graph(img, mask_labels, model, lambda, leftW, upleftW, upW, uprightW, graph, precomputed_contour_likelihood );
        estimateSegmentation( graph, mask_labels );

        //delete graph;
    }
    this->is_initialized = true;

    mask = LabelsFromMask(mask_labels);

    if (slack==cv::Rect())
        return mask;
    else
    {
        labels = cv::Mat::zeros(image_.size(), CV_8UC1);
        mask.copyTo(labels(slack));

        return labels;
    }
}