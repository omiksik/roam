#include "ROAM_KCF.h"

using namespace ROAM;

// -----------------------------------------------------------------------------------
cv::Rect KCF::Evaluate(const cv::Mat &img, Output &out, const cv::Rect &target)
// -----------------------------------------------------------------------------------
{
    // track?
    if(target.area() == 0)
        tracked_target = Track(img, out);

    // initialize or update
    if(target.area() > 0)
    {
        init(target); // create hann windows and labels
        train(img, padded_detection, true); // new model
    }
    else	
        train(img, padded_detection, false); // update

    return tracked_target;
}

// -----------------------------------------------------------------------------------
cv::Rect KCF::Evaluate(const cv::Mat &img, const cv::Rect &target)
// -----------------------------------------------------------------------------------
{
    // track?
    if(target.area() == 0)
        tracked_target = Track(img, this->output);

    // initialize or update
    if(target.area() > 0)
    {
        init(target); // create hann windows and labels
        train(img, padded_detection, true); // new model
    }
    else
        train(img, padded_detection, false); // update

    return tracked_target;
}

// -----------------------------------------------------------------------------------
KCF::Output KCF::GetCurrentOutput() const
// -----------------------------------------------------------------------------------
{
    return this->output;
}

// -----------------------------------------------------------------------------------
cv::Mat KCF::GetRegressionMap() const
// -----------------------------------------------------------------------------------
{
    return this->output.img_regression;
}

// -----------------------------------------------------------------------------------
cv::Rect KCF::Track(const cv::Mat &img, Output &out)
// -----------------------------------------------------------------------------------
{
    if(!initialized_model)
        throw std::logic_error("KCF::Track - regressor needs to be initialized first");

    // get patch
    const cv::Mat patch = getSubwindow(img, padded_detection );

    // get features
    const cv::Mat features = getFeatures(patch, params.feature_type, hann_window_detection);

    // patch fft
    const cv::Mat z_f = nonunitaryFFT(features);

    // kernel correlation kxz
    const cv::Mat kxz_f = evaluateKernelCorrelation(z_f, model.x_f, params.kernel);

    // fast detection
    const cv::Mat response_f = specMultiplication(model.alpha_f, kxz_f, false);
            
    // get response
    const cv::Mat complex_response = nonunitaryIFFT(response_f);

    cv::Mat m_response[2];
    cv::split(complex_response, m_response);

    // shift
    double min, max;
    cv::Point min_loc, delta;
    cv::minMaxLoc(m_response[0], &min, &max, &min_loc, &delta);

    // TODO: check 0 vs 1 indices
    // wrap around to negative half-space of vertical axis
    if(delta.y > z_f.rows / 2)
    {
        delta.y -= z_f.rows;
#ifdef _DEBUG
        std::cout << "[WARNING] - unwrapping delta.y, check whether it works" << std::endl;
#endif
    }

    if(delta.x > z_f.cols / 2)
    {
        delta.x -= z_f.cols;
#ifdef _DEBUG
        std::cout << "[WARNING] - unwrapping delta.x, check whether it works" << std::endl;
#endif
    }

    confidence = computePSR(m_response[0]);

    // return response
    {
        out.patch_translation = m_response[0];
        out.patch_regression = swapTranslation2Pixels(m_response[0]);
        out.img_regression = putSubwindow(img.size(), padded_detection, out.patch_regression);
        out.orig_tracked_target = tracked_target;
        out.orig_padded_target = padded_detection;
        out.patch_center = cv::Point2i(static_cast<int>(std::floor((out.patch_translation.cols - 1) / 2.0)), 
                                       static_cast<int>(std::floor((out.patch_translation.rows - 1) / 2.0)));
        out.orig_img_center = padded_detection.tl() + out.patch_center;
        out.PSR = confidence;
    }

    // TODO: again, check indices
    // update padded window
    padded_detection.y += delta.y;
    padded_detection.x += delta.x;

    // update returned response
    tracked_target.y += delta.y;
    tracked_target.x += delta.x;

    {
        out.shifted_tracked_target = tracked_target;
        out.shifted_padded_target = padded_detection;
        out.shifted_img_center = padded_detection.tl() + out.patch_center;
    }

    return tracked_target;
}

// -----------------------------------------------------------------------------------
void KCF::Update(const cv::Mat &img, const cv::Rect &target, const bool padded)
// -----------------------------------------------------------------------------------
{
    if(!initialized_model)
        throw std::logic_error("KCF::Update - regressor needs to be initialized first");

    if(padded && target.size() != padded_detection.size())
        throw std::invalid_argument("KCF::Update - dimensions of padded_detection and target must agree");
    
    if(!padded && target.size() != tracked_target.size())
        throw std::invalid_argument("KCF::Update - dimensions of tracked_object and target must agree");

    // compute delta
    const cv::Point2i delta = padded == true ? target.tl() - padded_detection.tl() : target.tl() - tracked_target.tl();

    // move rectangles
    padded_detection = adjustTarget(delta, padded_detection);
    tracked_target = adjustTarget(delta, tracked_target);

    // update
    if(confidence > params.train_confidence || !initialized_model)
        train(img, padded_detection, false);
}

// -----------------------------------------------------------------------------------
void KCF::init(const cv::Rect &target)
// -----------------------------------------------------------------------------------
{
    tracked_target = target;
        
    // hann for training
    const int extended_x_training = std::max(static_cast<int>(target.width * (1 + params.target_padding)), 
                                    2 * params.min_displacement.width);

    const int extended_y_training = std::max(static_cast<int>(target.height * (1 + params.target_padding)), 
                                    2 * params.min_displacement.height);

    cv::Rect padded_target;
    padded_target.x = target.x + static_cast<int>(std::floor(target.width / 2.0) - std::floor(extended_x_training / 2.0));
    padded_target.width = extended_x_training;

    padded_target.y = target.y + static_cast<int>(std::floor(target.height / 2.0) - std::floor(extended_y_training / 2.0));
    padded_target.height = extended_y_training;

    // hann for detection
    const int extended_x_detection = std::max(static_cast<int>(target.width * (1 + params.detection_padding)), extended_x_training);
    const int extended_y_detection = std::max(static_cast<int>(target.height * (1 + params.detection_padding)), extended_y_training);
        
    padded_detection.x = target.x + static_cast<int>(std::floor(target.width / 2.0) - std::floor(extended_x_detection / 2.0));
    padded_detection.width = extended_x_detection;

    padded_detection.y = target.y + static_cast<int>(std::floor(target.height / 2.0) - std::floor(extended_y_detection / 2.0));
    padded_detection.height = extended_y_detection;

    hann_window_detection = getHannWindow(padded_detection.size());
    hann_window_target = getHannWindow(padded_detection.size(), padded_target.size());
        
    // labels
    relative_sigma = static_cast<FLOAT_TYPE>(std::sqrt(target.area()) * params.target_sigma);
    const cv::Mat y = getGaussianShapedLabels(padded_detection.size(), relative_sigma);
    y_f = nonunitaryFFT(y);
}

// -----------------------------------------------------------------------------------
void KCF::train(const cv::Mat &img,
                                     const cv::Rect &target, 
                                     const bool initialize)
// -----------------------------------------------------------------------------------
{
    Model current;

    // get patch
    const cv::Mat patch = getSubwindow(img, target);

    // get features
    const cv::Mat features = getFeatures(patch, params.feature_type, hann_window_target);

    // patch fft
    current.x_f = nonunitaryFFT(features);	
    
    // kernel correlation kxx
    const cv::Mat k_f = evaluateKernelCorrelation(current.x_f, current.x_f, params.kernel);

    // compute coefficients, ie alpha_f = y_f ./ (k_f + lambda)
    current.alpha_f = specMultiplication(y_f, 1.0 / (k_f + params.lambda), false);

    // train or update
    if(!initialize)
        update(current, params.interp_patch, params.interp_alpha);
    else
    {
        model = current;
        initialized_model = true;
    }
}

// -----------------------------------------------------------------------------------
void KCF::update(const Model &current, const FLOAT_TYPE interp_patch,
                                      const FLOAT_TYPE interp_alpha)
// -----------------------------------------------------------------------------------
{
    if(!initialized_model)
        throw std::logic_error("KCF::update - regressor needs to be initialized first");

    assert(model.alpha_f.size == current.alpha_f.size);
    assert(model.x_f.size == current.x_f.size);

    model.alpha_f = (1.0 - interp_alpha) * model.alpha_f + interp_alpha * current.alpha_f;
    model.x_f = (1.0 - interp_patch) * model.x_f + interp_alpha * current.x_f;
}

// -----------------------------------------------------------------------------------
cv::Mat KCF::nonunitaryFFT(const cv::Mat &mat) const
// -----------------------------------------------------------------------------------
{
    // in theory, we should use sizes of pow(2)
    /*
    // calculate the size of DFT transform
    cv::Size dftSize;
    dftSize.width = cv::getOptimalDFTSize(mat.cols);
    dftSize.height = cv::getOptimalDFTSize(mat.rows);

    // allocate temporary buffers and initialize them with 0's
    cv::Mat tmp(dftSize, mat.type(), cv::Scalar::all(0));

    // copy mat to the top-left corners of tmp
    cv::Mat roi(tmp, cv::Rect(0, 0, mat.cols, mat.rows));
    mat.copyTo(tmp);
    */

    const size_t n_channels = mat.channels();
    cv::Mat mat_f = mat.clone();

    // now transform the padded mat in-place;
    // use "nonzeroRows" hint for faster processing
    if(n_channels == 1)
        cv::dft(mat_f, mat_f, cv::DFT_COMPLEX_OUTPUT, mat.rows);
    else
    {
        std::vector<cv::Mat> channels(n_channels);
        cv::split(mat_f, channels);

        #pragma omp parallel for
        for(auto i = 0; i < n_channels; ++i)
            cv::dft(channels[i], channels[i], cv::DFT_COMPLEX_OUTPUT, mat.rows);

        cv::merge(channels, mat_f);
    }

    //// extract values
    //cv::Mat mat_f;
    //mat_f(cv::Rect(0, 0, mat.cols, mat.rows)).copyTo(mat_f);

    return mat_f;
}

// -----------------------------------------------------------------------------------
cv::Mat KCF::nonunitaryIFFT(const cv::Mat &mat_f) const
// -----------------------------------------------------------------------------------
{
    // in theory, we should use sizes of pow(2)
    // calculate the size of DFT transform
    /*
    cv::Size dftSize;
    dftSize.width = cv::getOptimalDFTSize(mat_f.cols);
    dftSize.height = cv::getOptimalDFTSize(mat_f.rows);

    // allocate temporary buffers and initialize them with 0's
    cv::Mat tmp(dftSize, mat_f.type(), cv::Scalar::all(0));

    // copy mat to the top-left corners of tmp
    cv::Mat roi(tmp, cv::Rect(0, 0, mat_f.cols, mat_f.rows));
    mat.copyTo(tmp);
    */

    cv::Mat res;
    const size_t n_channels = mat_f.channels();

    // now transform the padded mat in-place;
    // use "nonzeroRows" hint for faster processing
    
    // this is weird we use scale for ifft but scale manually in kernels for direct fft
    if(n_channels == 2)
        cv::idft(mat_f, res, cv::DFT_SCALE, mat_f.rows);
    else
    {
        std::vector<cv::Mat> channels(n_channels);
        cv::split(mat_f, channels);

        #pragma omp parallel for
        for(auto i = 0; i < n_channels / 2; ++i)
        {
            // ONDRA: we should fix this...
            std::vector<cv::Mat> tmp(2);
            tmp[0] = channels[2 * i];
            tmp[1] = channels[2 * i + 1];

            cv::Mat one_mat;
            cv::merge(tmp, one_mat);

            cv::idft(one_mat, one_mat, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE, mat_f.rows);

            cv::split(one_mat, tmp);
            channels[2 * i] = tmp[0];
            channels[2 * i + 1] = tmp[1];
        }

        cv::merge(channels, res);
    }
    
    // extract values
    //cv::Mat mat_f;
    //tmp(cv::Rect(0, 0, mat.cols, mat.rows)).copyTo(mat_f);

    return res;
}

// -----------------------------------------------------------------------------------
cv::Mat KCF::evaluateKernelCorrelation(const cv::Mat &a_f,
                                       const cv::Mat &b_f, 
                                       const Kernels &type) const
// -----------------------------------------------------------------------------------
{
    assert(a_f.size == b_f.size);

    switch(type)
    {
        case KRR_LINEAR:
            return evaluateLinearKernel(a_f, b_f);
        case KRR_GAUSSIAN:
            return evaluateGaussianKernel(a_f, b_f);
        case KRR_POLYNOMIAL:
            return evaluatePolynomialKernel(a_f, b_f);
        default:
            throw std::invalid_argument("KCF::evaluateKernelCorrelation - unknown kernel");
    }
}

// -----------------------------------------------------------------------------------
cv::Mat KCF::evaluateLinearKernel(const cv::Mat &a_f, const cv::Mat &b_f) const
// -----------------------------------------------------------------------------------
{
    assert(a_f.channels() == b_f.channels());
    assert(a_f.size == b_f.size);

    cv::Mat k_f;
    const size_t n_channels = a_f.channels();

    if(n_channels <= 2)
        k_f = specMultiplication(a_f, b_f, true);
    else
    {
        std::vector<cv::Mat> channels_a(a_f.channels());
        std::vector<cv::Mat> channels_b(b_f.channels());
        cv::split(a_f, channels_a);
        cv::split(b_f, channels_b);

        k_f = cv::Mat::zeros(a_f.rows, a_f.cols, CV_32FC2);

        for(size_t i = 0; i < n_channels / 2; ++i)
        {
            std::vector<cv::Mat> tmp_a(2);
            tmp_a[0] = channels_a[2 * i];
            tmp_a[1] = channels_a[2 * i + 1];

            cv::Mat one_mat_a;
            cv::merge(tmp_a, one_mat_a);

            std::vector<cv::Mat> tmp_b(2);
            tmp_b[0] = channels_b[2 * i];
            tmp_b[1] = channels_b[2 * i + 1];

            cv::Mat one_mat_b;
            cv::merge(tmp_b, one_mat_b);

            k_f += specMultiplication(one_mat_a, one_mat_b, true);
        }
    }

    k_f /= static_cast<double>(k_f.total());

    return k_f;
}

// -----------------------------------------------------------------------------------
cv::Mat KCF::evaluateGaussianKernel(const cv::Mat &a_f, const cv::Mat &b_f) const
// -----------------------------------------------------------------------------------
{
    throw std::logic_error("KCF::evaluateGaussianKernel not implemented");
}

// -----------------------------------------------------------------------------------
cv::Mat KCF::evaluatePolynomialKernel(const cv::Mat &a_f, const cv::Mat &b_f) const
// -----------------------------------------------------------------------------------
{
    throw std::logic_error("KCF::evaluateGaussianKernel not implemented");
}

// -----------------------------------------------------------------------------------
cv::Mat KCF::getHannWindow(const cv::Size &sz, const cv::Size &target_sz) const
// -----------------------------------------------------------------------------------
{
    if(sz.area() < target_sz.area())
        throw std::invalid_argument("KCF::getHannWindow - target window cannot be larger than detection window");

    cv::Mat hann_x = cv::Mat::zeros(1, sz.width, CV_32FC1);
    cv::Mat hann_y = cv::Mat::zeros(sz.height, 1, CV_32FC1);

    int x1 = 0;
    int x2 = sz.width;
    int y1 = 0;
    int y2 = sz.height;

    if(target_sz.area() > 0)
    {
        cv::Point2i center(static_cast<int>(std::floor((sz.width - 1) / 2.0)), 
                           static_cast<int>(std::floor((sz.height - 1) / 2.0)));

        x1 = center.x - static_cast<int>(std::floor((target_sz.width - 1) / 2.0));
        x2 = x1 + target_sz.width;

        y1 = center.y - static_cast<int>(std::floor((target_sz.height - 1) / 2.0));
        y2 = y1 + target_sz.height;
    }

    #pragma omp parallel for
    for(int i = x1; i < x2; ++i)
        hann_x.at<float>(0, i) = static_cast<float>(0.5f * (1.0f - cos(2.0f * M_PI * (i - x1) / static_cast<FLOAT_TYPE>((x2 - x1) - 1.f))));

    #pragma omp parallel for
    for(int i = y1; i < y2; ++i)
        hann_y.at<float>(i, 0) = static_cast<float>(0.5f * (1.0f - cos(2.0f * M_PI * (i - y1) / static_cast<FLOAT_TYPE>((y2 - y1) - 1.0f))));
    
    const cv::Mat res = hann_y * hann_x;

    return res;
}

// -----------------------------------------------------------------------------------
cv::Mat KCF::getGaussianShapedLabels(const cv::Size &sz,
                                                          const FLOAT_TYPE sigma) const
// -----------------------------------------------------------------------------------
{
    //if(sz.height % 2 == 0 != sz.width % 2 == 0)
    //	std::cout << "warning - unclear center of patch" << std::endl;

    cv::Mat tmp = cv::Mat::zeros(sz, CV_32FC1);
    
    const cv::Point2i center(static_cast<int>(std::floor((sz.width - 1) / 2.0)), 
                             static_cast<int>(std::floor((sz.height - 1) / 2.0)));
    
    // compute gaussian
    const FLOAT_TYPE normalizer = -0.5f / (relative_sigma * relative_sigma);

    #pragma omp parallel for
    for(auto y = 0; y < tmp.rows; ++y)
    {
        const FLOAT_TYPE d_y = static_cast<FLOAT_TYPE>((y - center.y) * (y - center.y));
        for(int x = 0; x < tmp.cols; ++x)
        {
            const FLOAT_TYPE d_x = static_cast<FLOAT_TYPE>((x - center.x) * (x - center.x));
            tmp.at<float>(y, x) = std::exp(normalizer * (d_x + d_y));
        }
    }
    
    // shift into wraped-around corners
    cv::Mat labels = cv::Mat::zeros(tmp.rows, tmp.cols, CV_32FC1);

    // dst top-left part
    {
        cv::Rect src(center.x, center.y, tmp.cols - center.x, tmp.rows - center.y);
        cv::Rect dst(0, 0, tmp.cols - center.x, tmp.rows - center.y);
        cv::Mat tmp_src = tmp(src);
        cv::Mat labels_dst = labels(dst);

        tmp_src.copyTo(labels_dst);
    }

    // dst bottom-left part
    {
        cv::Rect src(center.x, 0, tmp.cols - center.x, center.y);
        cv::Rect dst(0, tmp.rows - center.y, tmp.cols - center.x, center.y);
        cv::Mat tmp_src = tmp(src);
        cv::Mat labels_dst = labels(dst);

        tmp_src.copyTo(labels_dst);
    }

    // dst bottom-right part
    {
        cv::Rect src(0, 0, center.x, center.y);
        cv::Rect dst(tmp.cols - center.x, tmp.rows - center.y, center.x, center.y);
        cv::Mat tmp_src = tmp(src);
        cv::Mat labels_dst = labels(dst);

        tmp_src.copyTo(labels_dst);
    }

    // dst top-right part
    {
        cv::Rect src(0, center.y, center.x, tmp.rows - center.y);
        cv::Rect dst(tmp.cols - center.x, 0, center.x, tmp.rows - center.y);
        cv::Mat tmp_src = tmp(src);
        cv::Mat labels_dst = labels(dst);

        tmp_src.copyTo(labels_dst);
    }

    // sanity check
    assert(labels.at<float>(0, 0) == 1);

    return labels;
}

// -----------------------------------------------------------------------------------
cv::Mat KCF::getFeatures(const cv::Mat &img,
                                              const Features &ft_type, 
                                              const cv::Mat &window) const
// -----------------------------------------------------------------------------------
{
    cv::Mat res;

    switch(ft_type)
    {
        case FT_GRAY:
            res = getGrayFeatures(img);
            break;
        case FT_COLOUR:
            res = getColourFeatures(img);
            break;
        default:
            throw std::logic_error("KCF::getFeatures - unknown features");
    }

    // apply window
    if(res.size == window.size)
    {
        assert(window.type() == CV_32FC1);
        
        const size_t n_channels = res.channels();

        std::vector<cv::Mat> channels(n_channels);
        cv::split(res, channels);
        
        for(size_t i = 0; i < n_channels; ++i)
            channels[i] = channels[i].mul(window);

        cv::merge(channels, res);
    }

    return res;
}

// -----------------------------------------------------------------------------------
cv::Mat KCF::getGrayFeatures(const cv::Mat &img) const
// -----------------------------------------------------------------------------------
{
    // convert to gray
    cv::Mat gray;
    if(img.type() == CV_8UC3)
        cv::cvtColor(img, gray, CV_BGR2GRAY);
    else if(img.channels() == 1)
        img.copyTo(gray);
    else
        throw std::invalid_argument("KCF::getGrayFeatures - requires either rgb or grayscale input");
    
    // do we have floats?
    if(gray.type() != CV_32FC1)
        gray.convertTo(gray, CV_32FC1, 1/255.0);

    // subtract mean
    const cv::Scalar mean = cv::mean(gray);
    gray -= mean;

    // L2 normalize ???
    // it might be better...

    return gray;
}

// -----------------------------------------------------------------------------------
cv::Mat KCF::getColourFeatures(const cv::Mat &img) const
// -----------------------------------------------------------------------------------
{
    if(img.channels() != 3)
        throw std::invalid_argument("[WARNING] KCF::getColourFeatures - colour features should have 3 channels");

    cv::Mat features;
    
    if(img.type() != CV_32FC3)
        img.convertTo(features, CV_32FC1, 1/255.0);

    const cv::Scalar mean = cv::mean(features);
    features -= mean;

    // L2 normalize ???
    // it might be better...

    return features;
}

// -----------------------------------------------------------------------------------
cv::Mat KCF::getSubwindow(const cv::Mat &img, const cv::Rect &rect) const
// -----------------------------------------------------------------------------------
{
    // TODO check whether at least one pixel is within the image, crashes otherwise

    const int top = std::abs(std::min(0, rect.y));
    const int left = std::abs(std::min(0, rect.x));
    const int bottom = std::abs(std::max(0, rect.y + rect.height - img.rows));
    const int right = std::abs(std::max(0, rect.x + rect.width - img.cols));

    cv::Mat res = cv::Mat::zeros(rect.size(), img.type());

    // make sure we extract only valid pixels
    const cv::Rect subrect = rect & cv::Rect(0, 0, img.cols, img.rows);
    cv::Mat submat = img(subrect);

    // copy into the middle and replicate borders if needed
    cv::copyMakeBorder(submat, res, top, bottom, left, right, cv::BORDER_REPLICATE);

    return res;
}

// -----------------------------------------------------------------------------------
cv::Mat KCF::putSubwindow(const cv::Size &img_size,
                                               const cv::Rect &rect, 
                                               const cv::Mat &subpart) const
// -----------------------------------------------------------------------------------
{
    // make sure we extract only valid pixels
    const int top = std::max(0, 0 - rect.y);
    const int left = std::max(0, 0 - rect.x);
    const int bottom = std::max(0, rect.y + rect.height - img_size.height);
    const int right = std::max(0, rect.x + rect.width - img_size.width);

    const cv::Rect srcRect(left, top, subpart.cols - right - left, subpart.rows - bottom - top);
    const cv::Rect valid_rect = rect & cv::Rect(0, 0, img_size.width, img_size.height);

    cv::Mat res = cv::Mat::zeros(img_size, subpart.type());

    cv::Mat src = subpart(srcRect);
    cv::Mat dst = res(valid_rect);

    src.copyTo(dst);

    return res;
}

// -----------------------------------------------------------------------------------
cv::Mat KCF::specMultiplication(const cv::Mat &a_f,
                                                     const cv::Mat &b_f, 
                                                     const bool conjB) const
// -----------------------------------------------------------------------------------
{
    assert(a_f.size == b_f.size);
    assert(a_f.type() == b_f.type());

    // get real and imag matrices
    cv::Mat a_f_sep[2];
    cv::Mat b_f_sep[2];

    cv::split(a_f, a_f_sep);
    cv::split(b_f, b_f_sep);

    // do we want to conjugate B?
    if(conjB)
        b_f_sep[1] = -b_f_sep[1];

    // element-wise complex multiplication
    const cv::Mat real0 = a_f_sep[0].mul(b_f_sep[0]);
    const cv::Mat real1 = a_f_sep[1].mul(b_f_sep[1]);
    const cv::Mat imag0 = a_f_sep[0].mul(b_f_sep[1]);
    const cv::Mat imag1 = a_f_sep[1].mul(b_f_sep[0]);

    std::vector<cv::Mat> mult(2);
    
    mult[0] = real0 - real1;
    mult[1] = imag0 + imag1;;

    // return back as a single matrix
    cv::Mat res;
    cv::merge(mult, res);

    return res;
}


// -----------------------------------------------------------------------------------
cv::Mat KCF::swapTranslation2Pixels(const cv::Mat &translations) const
// -----------------------------------------------------------------------------------
{
    const cv::Point2i center(static_cast<int>(std::floor((translations.cols - 1) / 2.0)), 
                             static_cast<int>(std::floor((translations.rows - 1) / 2.0)));
    
    cv::Mat response = cv::Mat::zeros(translations.rows, translations.cols, CV_32FC1);

#if 0
        if(delta.y > z_f.rows / 2) delta.y -= z_f.rows;
        if(delta.x > z_f.cols / 2) delta.x -= z_f.cols;

        padded_detection.y += delta.y;
        padded_detection.x += delta.x;
#endif

    // src top-left part
    {
        cv::Rect dst(center.x, center.y, translations.cols - center.x, translations.rows - center.y);
        cv::Rect src(0, 0, translations.cols - center.x, translations.rows - center.y);
        cv::Mat tmp_src = translations(src);
        cv::Mat labels_dst = response(dst);

        tmp_src.copyTo(labels_dst);
    }

    // src bottom-left part
    {
        cv::Rect dst(center.x, 0, translations.cols - center.x, center.y);
        cv::Rect src(0, translations.rows - center.y, translations.cols - center.x, center.y);
        cv::Mat tmp_src = translations(src);
        cv::Mat labels_dst = response(dst);

        tmp_src.copyTo(labels_dst);
    }

    // src bottom-right part
    {
        cv::Rect dst(0, 0, center.x, center.y);
        cv::Rect src(translations.cols - center.x, translations.rows - center.y, center.x, center.y);
        cv::Mat tmp_src = translations(src);
        cv::Mat labels_dst = response(dst);

        tmp_src.copyTo(labels_dst);
    }

    // src top-right part
    {
        cv::Rect dst(0, center.y, center.x, translations.rows - center.y);
        cv::Rect src(translations.cols - center.x, 0, center.x, translations.rows - center.y);
        cv::Mat tmp_src = translations(src);
        cv::Mat labels_dst = response(dst);

        tmp_src.copyTo(labels_dst);
    }

    return response;
}

// -----------------------------------------------------------------------------------
cv::Rect KCF::adjustTarget(const cv::Point2i &delta, const cv::Rect &rect)
// -----------------------------------------------------------------------------------
{
    cv::Rect adjusted = rect;
    adjusted.x += delta.x;
    adjusted.y += delta.y;

    return adjusted;
}

// -----------------------------------------------------------------------------------
cv::Rect KCF::adjustTarget(const cv::Point2i &orig_center, const cv::Point2i &max_response, const cv::Rect &rect)
// -----------------------------------------------------------------------------------
{
    cv::Rect adjusted = rect;

    cv::Point2i delta = max_response - orig_center;

    adjusted.x += delta.x;
    adjusted.y += delta.y;

    return adjusted;
}

// -----------------------------------------------------------------------------------
cv::Rect KCF::adjustTarget(const cv::Point2i &orig_center, const cv::Mat &patch, const cv::Rect &rect)
// -----------------------------------------------------------------------------------
{
    cv::Rect adjusted = rect;

    double min, max;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(patch, &min, &max, &min_loc, &max_loc);

    cv::Point2i delta = max_loc - orig_center;

    adjusted.x += delta.x;
    adjusted.y += delta.y;

    return adjusted;
}

// -----------------------------------------------------------------------------------
cv::Size KCF::getTargetSize() const
// -----------------------------------------------------------------------------------
{
    return tracked_target.size();
}
    
// -----------------------------------------------------------------------------------
cv::Size KCF::getWindowSize() const
// -----------------------------------------------------------------------------------
{
    return padded_detection.size();
}

// -----------------------------------------------------------------------------------
cv::Rect KCF::getTrackedTarget() const
// -----------------------------------------------------------------------------------
{
    return tracked_target;
}

// -----------------------------------------------------------------------------------
cv::Rect KCF::getPaddedTarget() const
// -----------------------------------------------------------------------------------
{
    return padded_detection;
}

// -----------------------------------------------------------------------------------
cv::Point2i KCF::getCurrentReferencePoint() const
// -----------------------------------------------------------------------------------
{
    const cv::Size sz = padded_detection.size();
    const cv::Point2i center(static_cast<int>(std::floor((sz.width - 1) / 2.0)), static_cast<int>(std::floor((sz.height - 1) / 2.0)));
    return padded_detection.tl() + center;
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE KCF::computePSR(const cv::Mat& response, const int half_window) const
// -----------------------------------------------------------------------------------
{
    if (response.type() != CV_32FC1)
        throw std::invalid_argument("KCF::computePSR - response should be single channel float");

    double min, max;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(response, &min, &max, &min_loc, &max_loc);

    const cv::Rect peak_rect(max_loc.x - half_window, max_loc.y - half_window, 2 * half_window + 1, 2 * half_window + 1);

    // TODO: this should be re-implemented

    FLOAT_TYPE mean = 0.0;
    size_t n_pixels = 0;
    for (int y = 0; y < response.rows; ++y)
        for (int x = 0; x < response.cols; ++x)
            if (x > peak_rect.tl().x && x < peak_rect.br().x && y > peak_rect.tl().y && y < peak_rect.br().y)
                continue;
            else
            {
                mean += response.at<float>(y, x);
                n_pixels++;
            }

    mean /= static_cast<FLOAT_TYPE>(n_pixels);

    FLOAT_TYPE var = 0.0;
    for (int y = 0; y < response.rows; ++y)
        for (int x = 0; x < response.cols; ++x)
            if (x > peak_rect.tl().x && x < peak_rect.br().x && y > peak_rect.tl().y && y < peak_rect.br().y)
                continue;
            else
                var += std::pow(response.at<float>(y, x) - mean, 2);

    var /= static_cast<FLOAT_TYPE>(n_pixels);

    const FLOAT_TYPE std = std::sqrt(var);
    const FLOAT_TYPE PSR = static_cast<FLOAT_TYPE>((max - mean) / std);

    return PSR;
};
