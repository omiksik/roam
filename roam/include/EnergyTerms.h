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

#include "Configuration.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#ifdef HAVE_OPENCV_CONTRIB
    #include <opencv2/ximgproc.hpp>
#endif

//-----------------------------------------------------------------------------------------------------------------------
//----------------------------------------------- DEFINITIONS -----------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------

#define BOILERPLATE_CODE_UNARY(name,classname) \
    static cv::Ptr<classname> createUnaryTerm(const classname::Params &parameters=classname::Params());\
    virtual ~classname(){};

#define BOILERPLATE_CODE_PAIRWISE(name,classname) \
    static cv::Ptr<classname> createPairwiseTerm(const classname::Params &parameters=classname::Params());\
    virtual ~classname(){};

namespace ROAM
{
//-----------------------------------------------------------------------------------------------------------------------
//----------------------------------------------- BASE CLASSES ----------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------

/*!
 * \brief The UnaryTerm class is a base class for all possible Unary terms.
 */
// -----------------------------------------------------------------------------------
class UnaryTerm : public cv::Algorithm
// -----------------------------------------------------------------------------------
{
public:
    explicit UnaryTerm();
    virtual ~UnaryTerm();

    /*!
     * \brief Initialized the Unary Term.
     * \param image
     * \return true is everuthing went alright during initialization
     */
    bool Init(const cv::Mat &image);

    /*!
     * \brief updates the Unary Term.
     * \param image
     * \return
     */
    bool Update(const cv::Mat &image);

    /*!
     * \brief GetUnariesDimension
     * \return
     */
    cv::Size GetUnariesDimension() const;

    /*!
     * \brief GetUnaries
     * \return
     */
    cv::Mat GetUnaries() const;

    /*!
     * \brief getCost
     * \param coordinates
     */
    virtual FLOAT_TYPE GetCost(const cv::Point &coordinates) = 0;

    /*!
     * \brief create a Unary Term by its name.
     * \param unaryType
     * \return
     */
    static cv::Ptr<UnaryTerm> create(const cv::String &unaryType);

    /*!
     * \brief IsInitialized
     * \return
     */
    bool IsInitialized() const;

    virtual FLOAT_TYPE GetWeight() const = 0;

protected:
    bool is_initialized;
    cv::Mat unaries; //!< This is the actual unary cost for all possible coordinates
                     // Observe that it is NOT a static element.

    virtual bool initImpl(const cv::Mat &image) = 0;
    virtual bool updateImpl(const cv::Mat &image) = 0;
};


/*!
 * \brief The PairwiseTerm base class for all possible Pairwise terms.
 */
// -----------------------------------------------------------------------------------
class PairwiseTerm : public cv::Algorithm
// -----------------------------------------------------------------------------------
{
public:
    explicit PairwiseTerm();

    virtual ~PairwiseTerm();

    /*!
     * \brief init
     * \param matrix_a
     * \param matrix_b
     * \return
     */
    bool Init(const cv::Mat &matrix_a, const cv::Mat &matrix_b);

    /*!
     * \brief update
     * \param matrix_a
     * \param matrix_b
     * \return
     */
    bool Update(const cv::Mat &matrix_a, const cv::Mat &matrix_b);

    /*!
     * \brief IsInitialized
     * \return
     */
    bool IsInitialized() const;

    /*!
     * \brief getCost
     * \param coordinate_a
     * \param coordinate_b
     * \return
     */
    virtual FLOAT_TYPE GetCost(const cv::Point &coordinate_a, const cv::Point &coordinate_b) = 0;

    /*!
     * \brief GetCosts is intended to allow possible paralell implementations of GetCost
     * \param coordinates_a
     * \param coordinates_b
     * \param costs
     */
    virtual void GetCosts(const std::vector<cv::Point> &coordinates_a, const std::vector<cv::Point> &coordinates_b,
                          std::vector<FLOAT_TYPE> &costs);

    static cv::Ptr<PairwiseTerm> create(const cv::String &pairwiseType);

    virtual FLOAT_TYPE GetWeight() const = 0;

protected:
    bool is_initialized;
    cv::Mat matrix_a; //!< We assume that two matrices are needed to compute the pairwise terms
    cv::Mat matrix_b; //!< The second matrix.

    virtual bool initImpl(const cv::Mat &matrix_a, const cv::Mat &matrix_b) = 0;
    virtual bool updateImpl(const cv::Mat &matrix_a, const cv::Mat &matrix_b) = 0;
};

//-----------------------------------------------------------------------------------------------------------------------
//----------------------------------------------- DERIV CLASSES ---------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
class GradientUnary : public UnaryTerm
// -----------------------------------------------------------------------------------
{
public:

    /*!
     * \brief The GradType enum
     */
    // -------------------------------------------------------------------------------
    enum GradType 
    // -------------------------------------------------------------------------------
    {
        SOBEL = 601,
        SCHARR,
        LAPLACIAN,
        PIOTR_FORESTS
    };

    /*!
     * \brief The Params struct
     */
    // -------------------------------------------------------------------------------
    struct Params
    // -------------------------------------------------------------------------------
    {
        // ---------------------------------------------------------------------------
        explicit Params(const GradType grad_type = GradType::SOBEL, 
                        const int kernel_size = 3, 
                        const FLOAT_TYPE weight = 1.0, 
                        const FLOAT_TYPE smooth_factor = 5,
                        const std::string &model_name = std::string())
        // ---------------------------------------------------------------------------
        {
            this->grad_type = grad_type;
            this->kernel_size = kernel_size;
            this->weight = weight;
            this->smooth_factor = smooth_factor;
            this->model_name_trained_detector = model_name;
        }

        void Read(const cv::FileNode& /*fn*/);
        void Write(cv::FileStorage & /*fs*/) const;

        GradType grad_type; //!< Sobel, Scharr or Laplacian
        int kernel_size;    //!< Kernel size for the convolutional kernel (Only applies for Sobel and Laplacian)
        FLOAT_TYPE weight;
        FLOAT_TYPE smooth_factor;
        std::string model_name_trained_detector;
    };

    /*!
     * \brief Constructor
     * \param parameters of GradientUnary
     */
    BOILERPLATE_CODE_UNARY("GRAD", GradientUnary)

protected:
    Params params;
};

// -----------------------------------------------------------------------------------
class GradientDTUnary : public UnaryTerm
// -----------------------------------------------------------------------------------
{
public:

    /*!
     * \brief The GradType enum
     */
    // -------------------------------------------------------------------------------
    enum GradType
    // -------------------------------------------------------------------------------
    {
        SOBEL = 601,
        SCHARR,
        LAPLACIAN,
        PIOTR_FORESTS
    };

    /*!
     * \brief The Params struct
     */
    // -------------------------------------------------------------------------------
    struct Params
    // -------------------------------------------------------------------------------
    {
        // ---------------------------------------------------------------------------
        explicit Params(const GradType grad_type = GradType::SOBEL,
                        const int kernel_size = 3,
                        const FLOAT_TYPE weight = 1.0,
                        const FLOAT_TYPE smooth_factor = 5,
                        const std::string &model_name = std::string())
        // ---------------------------------------------------------------------------
        {
            this->grad_type = grad_type;
            this->kernel_size = kernel_size;
            this->weight = weight;
            this->smooth_factor = smooth_factor;
            this->model_name_trained_detector = model_name;
        }

        void Read(const cv::FileNode& /*fn*/);
        void Write(cv::FileStorage & /*fs*/) const;

        GradType grad_type; //!< Sobel, Scharr or Laplacian
        int kernel_size;    //!< Kernel size for the convolutional kernel (Only applies for Sobel and Laplacian)
        FLOAT_TYPE weight;
        FLOAT_TYPE smooth_factor;
        std::string model_name_trained_detector;
    };

    /*!
     * \brief Constructor
     * \param parameters of GradientUnary
     */
    BOILERPLATE_CODE_UNARY("GRADDT", GradientDTUnary)

protected:
    Params params;
};

// -----------------------------------------------------------------------------------
class GenericUnary : public UnaryTerm
// -----------------------------------------------------------------------------------
{
public:

    /*!
     * \brief The PoolingType enum
     */
    // -------------------------------------------------------------------------------
    enum PoolingType
    // -------------------------------------------------------------------------------
    {
        NO_POOL = 801,
        MAX_POOL,
        MEAN_POOL
    };

    /*!
     * \brief The Params struct
     */
    // -------------------------------------------------------------------------------
    struct Params
    // -------------------------------------------------------------------------------
    {
        // ---------------------------------------------------------------------------
        explicit Params(const FLOAT_TYPE weight = 1.f,
                        const PoolingType pooling_type = NO_POOL,
                        const FLOAT_TYPE pooling_win_size = 1.f)
        // ---------------------------------------------------------------------------
        {
            this->weight = weight;
            this->pooling_type= pooling_type;
            this->pooling_win_size = pooling_win_size;
        }

        PoolingType pooling_type; //!< MEAN OR MAX
        FLOAT_TYPE weight;
        FLOAT_TYPE pooling_win_size;
    };

    /*!
     * \brief Constructor
     * \param parameters of genericUnary
     */
    BOILERPLATE_CODE_UNARY("GENE", GenericUnary)

protected:
    Params params;
};

// -----------------------------------------------------------------------------------
class DistanceUnary : public UnaryTerm
// -----------------------------------------------------------------------------------
{
public:

    /*!
     * \brief The Params struct
     */
    // -------------------------------------------------------------------------------
    struct Params
    // -------------------------------------------------------------------------------
    {
        // ---------------------------------------------------------------------------
        explicit Params(const FLOAT_TYPE weight = 1.f)
        // ---------------------------------------------------------------------------
        {
            this->weight = weight;
        }

        FLOAT_TYPE weight;
    };

    void SetPastDistance(const FLOAT_TYPE dist);
    FLOAT_TYPE GetPastDistance() const;

    /*!
     * \brief Constructor
     * \param parameters of genericUnary
     */
    BOILERPLATE_CODE_UNARY("DIST", DistanceUnary)

protected:
    Params params;
    FLOAT_TYPE past_distance;
};

// -----------------------------------------------------------------------------------
class NormPairwise : public PairwiseTerm
// -----------------------------------------------------------------------------------
{
public:
    /*!
     * \brief The NormType enum
     */
    // -------------------------------------------------------------------------------
    enum NormType {
    // -------------------------------------------------------------------------------
        L2_SQR = 501,
        L2,
        LINF,
        L1,
        L0
    };

    // -------------------------------------------------------------------------------
    struct Params
    // -------------------------------------------------------------------------------
    {
        // ---------------------------------------------------------------------------
        explicit Params(const NormType norm_type = NormType::L2_SQR, 
                        const FLOAT_TYPE weight = 1.0)
        // ---------------------------------------------------------------------------
        {
            this->norm_type = norm_type;
            this->weight = weight;
        }

        void Read(const cv::FileNode& /*fn*/ );
        void Write(cv::FileStorage & /*fs*/ ) const;

        NormType norm_type; //!<
        FLOAT_TYPE weight;
    };

    /*!
     * \brief Constructor
     * \param parameters of NormPairwise
     */
    BOILERPLATE_CODE_PAIRWISE("NORM", NormPairwise)

protected:
    Params params;
};

// -----------------------------------------------------------------------------------
class GenericPairwise : public PairwiseTerm
// -----------------------------------------------------------------------------------
{
public:

    // -------------------------------------------------------------------------------
    struct Params
    // -------------------------------------------------------------------------------
    {
        // ---------------------------------------------------------------------------
        explicit Params(const FLOAT_TYPE weight = 1.0)
        // ---------------------------------------------------------------------------
        {
            this->weight = weight;
        }
        FLOAT_TYPE weight;
    };

    /*!
     * \brief Constructor
     * \param parameters of NormPairwise
     */
    BOILERPLATE_CODE_PAIRWISE("GENE", GenericPairwise)

protected:
    Params params;
};

// -----------------------------------------------------------------------------------
class TempNormPairwise : public PairwiseTerm
// -----------------------------------------------------------------------------------
{
public:
    /*!
     * \brief The TempNormType enum
     */
    // -------------------------------------------------------------------------------
    struct Params
    // -------------------------------------------------------------------------------
    {
        // ---------------------------------------------------------------------------
        explicit Params(const FLOAT_TYPE weight = 1.0)
        // ---------------------------------------------------------------------------
        {
            this->weight=weight;
        }

        void Read(const cv::FileNode& /*fn*/ ){}
        void Write(cv::FileStorage & /*fs*/ ) const{}

        FLOAT_TYPE weight;
    };

    /*!
     * \brief Constructor
     * \param parameters of TempNormPairwise
     */
    BOILERPLATE_CODE_PAIRWISE("TEMPNORM", TempNormPairwise)

protected:
    Params params;
};

// -----------------------------------------------------------------------------------
class TempAnglePairwise : public PairwiseTerm
// -----------------------------------------------------------------------------------
{
public:
    /*!
     * \brief The TempAngleType enum
     */
    // -------------------------------------------------------------------------------
    struct Params
    // -------------------------------------------------------------------------------
    {
        // ---------------------------------------------------------------------------
        explicit Params(const FLOAT_TYPE weight = 1.0)
        // ---------------------------------------------------------------------------
        {
            this->weight=weight;
        }

        void Read(const cv::FileNode& /*fn*/ ){}
        void Write(cv::FileStorage & /*fs*/ ) const{}

        FLOAT_TYPE weight;
    };

    /*!
     * \brief Constructor
     * \param parameters of TempAnglePairwise
     */
    BOILERPLATE_CODE_PAIRWISE("TEMPANGLE", TempAnglePairwise)

protected:
    Params params;
};

} // namespace roam
