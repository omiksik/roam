#pragma once
//#include <Eigen/StdVector>
//EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix2d)
#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>

class Eigen2OpenCVUtils
{
public:
	// -----------------------------------------------------------------------------------
	static Eigen::MatrixXd opencvMat2Eigen(const cv::Mat &mat)
	// -----------------------------------------------------------------------------------
	{
		Eigen::MatrixXd res = Eigen::MatrixXd::Zero(mat.rows, mat.cols);

		#pragma omp parallel for
		for(auto y = 0; y < mat.rows; ++y)
			for(auto x = 0; x < mat.cols; ++x)
				res(y, x) = mat.at<double>(y, x);

		return res;
	}

	// -----------------------------------------------------------------------------------
	static cv::Mat eigen2opencv(const Eigen::MatrixXd &mat)
	// -----------------------------------------------------------------------------------
	{
		cv::Mat res(static_cast<int>(mat.rows()), static_cast<int>(mat.cols()), CV_64FC1);
		#pragma omp parallel for
		for(auto y = 0; y < mat.rows(); ++y)
			for(auto x = 0; x < mat.cols(); ++x)
				res.at<double>(y, x) = mat(y, x);

		return res;
	}

	// -----------------------------------------------------------------------------------
	static cv::Mat eigen2OpenCV(const Eigen::VectorXi &vct, const size_t n_rows, const size_t n_labels)
	// -----------------------------------------------------------------------------------
	{
		const size_t n_cols = vct.size() / n_rows;
		cv::Mat res(static_cast<int>(n_rows), static_cast<int>(n_cols), CV_8UC3);

		#pragma omp parallel for
		for(auto y = 0; y < n_rows; ++y)
			for(auto x = 0; x < n_cols; ++x)
			{
				res.at<cv::Vec3b>(y, x)[0] = static_cast<uchar>(255.0 * vct(y * n_cols + x) / static_cast<double>(n_labels));
				res.at<cv::Vec3b>(y, x)[1] = static_cast<uchar>(255.0 * vct(y * n_cols + x) / static_cast<double>(n_labels));
				res.at<cv::Vec3b>(y, x)[2] = static_cast<uchar>(255.0 * vct(y * n_cols + x) / static_cast<double>(n_labels));
			}

		return res;
	}
};

class OpenCVUtils
{
public:
	// -----------------------------------------------------------------------------------
	template <typename T>
	static cv::Mat replaceRow(const cv::Mat &mat, const std::vector<T> &vct, const size_t row)
	// -----------------------------------------------------------------------------------
	{
		if(mat.rows < row)
			throw std::invalid_argument("OpenCVUtils::swapRow - mat.rows < row");

		if(vct.size() != mat.cols)
			throw std::invalid_argument("OpenCVUtils::swapRow - size(vct) != mat.cols");

		cv::Mat tmp;
		cv::transpose(cv::Mat(vct, false), tmp);

		tmp.row(0).copyTo(mat.row(static_cast<int>(row)));
	
		return mat;
	}

	// -----------------------------------------------------------------------------------
	template <typename T>
	static cv::Mat vct2row(const std::vector<T> &vct)
	// -----------------------------------------------------------------------------------
	{
		cv::Mat res;
		cv::transpose(cv::Mat(vct, false), res);

		return res;
	}

	// -----------------------------------------------------------------------------------
	template <typename T>
	static cv::Mat vct2column(const std::vector<T> &vct)
	// -----------------------------------------------------------------------------------
	{
		return cv::Mat(vct, false);
	}

	// -----------------------------------------------------------------------------------
	template <typename T>
	static std::vector<T> matRow2vct(const cv::Mat &mat, const size_t row = 0, 
									 const size_t n_elements = 0)
	// -----------------------------------------------------------------------------------
	{
		if(mat.rows < row)
			throw std::invalid_argument("OpenCVUtils::matRow2vct - mat.rows < row");

		const size_t n = n_elements == 0 ? mat.cols : n_elements;

		if(n > mat.cols)
			throw std::invalid_argument("OpenCVUtils::matRow2vct - n_elements > mat.columns");

		std::vector<T> vct;
		if(mat.type() == CV_64FC1)
			vct = std::vector<T>(mat.ptr<double>(static_cast<int>(row)), mat.ptr<double>(static_cast<int>(row))+n);
		else if(mat.type() == CV_32FC1)
			vct = std::vector<T>(mat.ptr<float>(static_cast<int>(row)), mat.ptr<float>(static_cast<int>(row))+n);
		else if(mat.type() == CV_8UC1)
			vct = std::vector<T>(mat.ptr<uchar>(static_cast<int>(row)), mat.ptr<uchar>(static_cast<int>(row))+n);
		else
			throw std::invalid_argument("OpenCVUtils::matRow2vct - unknown data type");

		return vct;
	}

	// -----------------------------------------------------------------------------------
	template <typename T>
	static std::vector<T> matColumn2vct(const cv::Mat &mat, const size_t col = 0)
	// -----------------------------------------------------------------------------------
	{
		if(mat.cols < col)
			throw std::invalid_argument("OpenCVUtils::matRow2vct - mat.rows < row");

		cv::Mat mat_t;
		cv::transpose(mat, mat_t);

		const std::vector<T> vct = matRow2vct<T>(mat_t, col);

		return vct;
	}
	
	// -----------------------------------------------------------------------------------
	static cv::Mat vct2mat(const std::vector<double> &vct, const size_t height)
	// -----------------------------------------------------------------------------------
	{
		const size_t width = vct.size() / height;
		cv::Mat res = cv::Mat(static_cast<int>(height), static_cast<int>(width), CV_64FC1, const_cast<double*>(vct.data()));
		return res;
	}

	// -----------------------------------------------------------------------------------
	static cv::Mat vct2mat(const std::vector<uchar> &vct, const size_t height)
	// -----------------------------------------------------------------------------------
	{
		const size_t width = vct.size() / (3 * height);
		cv::Mat res = cv::Mat(static_cast<int>(height), static_cast<int>(width), CV_8UC3, const_cast<uchar*>(vct.data()));
		return res;
	}

	// -----------------------------------------------------------------------------------
	static cv::Mat vct2mat(const std::vector<int> &vct, const size_t height)
	// -----------------------------------------------------------------------------------
	{
		const size_t width = vct.size() / height;
		cv::Mat res = cv::Mat(static_cast<int>(height), static_cast<int>(width), cv::DataType<int>::type, const_cast<int*>(vct.data()));
		return res;
	}

	// -----------------------------------------------------------------------------------
	static void RGB2BGR(cv::Mat &inOut)
	// -----------------------------------------------------------------------------------
	{
		// cvtcolor instead?

		cv::Mat ch1, ch2, ch3;
		// "channels" is a vector of 3 Mat arrays:
		std::vector<cv::Mat> channels(3);
		// split img:
		cv::split(inOut, channels);

		std::vector<cv::Mat> channels2(3);
		channels2[0] = channels[2];
		channels2[1] = channels[1];
		channels2[2] = channels[0];

		cv::merge(channels2, inOut);
	}

	// -----------------------------------------------------------------------------------
	static cv::Mat RGB2BGR(const cv::Mat &in)
	// -----------------------------------------------------------------------------------
	{
		cv::Mat inOut = in.clone();
		RGB2BGR(inOut);

		return inOut;
	}
};

// -----------------------------------------------------------------------------------
class VisualizationUtils
// -----------------------------------------------------------------------------------
{
public:

	// -----------------------------------------------------------------------------------
	static cv::Mat extractMapFromCosts(const Eigen::VectorXd &unary, 
									   const size_t height, const size_t width)
    // -----------------------------------------------------------------------------------
	{
		const size_t n_labels = unary.size() / (width * height);

		cv::Mat res = cv::Mat::zeros(static_cast<int>(height), static_cast<int>(width), CV_8UC1);

		#pragma omp parallel for
		for(auto y = 0; y < height; ++y)
			for(auto x = 0; x < width; ++x)
			{
				const size_t idx = (y * width + x) * n_labels;

				const Eigen::VectorXd block = unary.segment(idx, n_labels);

				Eigen::VectorXd::Index minEl;
				block.minCoeff(&minEl);

				res.at<uchar>(y, x) = static_cast<uchar>(minEl);
			}

		return res;
	}

	// -----------------------------------------------------------------------------------
	static std::vector<unsigned short> extractVctMapFromCosts(const Eigen::VectorXd &unary,
									   const size_t height, const size_t width)
	// -----------------------------------------------------------------------------------
	{
		const cv::Mat mat = extractMapFromCosts(unary, height, width);

		std::vector<unsigned short> res(mat.data, mat.data + (mat.cols * mat.rows)); //mat.dataend);// + (mat.cols * mat.rows));;

		return res;
	}

	// -----------------------------------------------------------------------------------
	static cv::Mat colorizeMAP(const cv::Mat &mat, const size_t n_classes)
	// -----------------------------------------------------------------------------------
	{
		if(n_classes < 2)
			throw std::invalid_argument("VisualizationUtils::colorizeMap - n_classes must be >= 2");

		cv::Mat colorized = mat * 1.0 / static_cast<double>(n_classes - 1);
		return colorized;
	}

	// -----------------------------------------------------------------------------------
	static cv::Mat colorize(const cv::Mat &img, const int colormap = cv::COLORMAP_JET,
							const bool expand_range = false, const bool disp_range = true)
	// -----------------------------------------------------------------------------------
	{
		double min, max;
		cv::minMaxLoc(img, &min, &max);

		cv::Mat adjusted_img;

		if(expand_range)
		{
			cv::normalize(img, adjusted_img, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		}
		else
		{
			adjusted_img = img * 255.0;
			adjusted_img.convertTo(adjusted_img, CV_8UC1);
		}

		cv::Mat output;
		cv::applyColorMap(adjusted_img, output, colormap);

		if(disp_range)
		{
			const std::string text = "min value: " + std::to_string(min) + ", max value: " + std::to_string(max);
			cv::putText(output, text, cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.25, cv::Scalar(0, 0, 255), 1, CV_AA);
			if(expand_range)
				cv::putText(output, "(maximized contrast)", cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.25, cv::Scalar(0, 0, 255), 1, CV_AA);
		}

		return output;
	}

	// -----------------------------------------------------------------------------------
	static void plotColormap(const std::string &window_name, const cv::Mat &img,
							 const int colormap = cv::COLORMAP_JET, const bool expand_range = false,
							 const size_t wait_period = 1)
	// -----------------------------------------------------------------------------------
	{
		cv::Mat mat = colorize(img, colormap, expand_range);
		cv::imshow(window_name, mat);
		cv::waitKey(static_cast<int>(wait_period));
	}

	// -----------------------------------------------------------------------------------
	template <typename T>
	static cv::Mat unary2mat(const std::vector<T> &unary, const size_t height, 
							 const size_t width, const size_t label_id, 
							 const int colormap = cv::COLORMAP_JET,
							 const std::string &window_name = std::string(),
							 const bool maximize_contrast = false)
	// -----------------------------------------------------------------------------------
	{
		const size_t n_classes = unary.size() / (width * height);

		if(n_classes < label_id)
			throw std::invalid_argument("VisualizationUtils::unary2mat - n_classes < label_id");

		cv::Mat mat(height, width, CV_64FC1);

		#pragma omp parallel for
		for(int y = 0; y < height; ++y)
			for(int x = 0; x < width; ++x)
				mat.at<double>(y, x) = unary[(y * width + x) * n_classes + label_id];

		const cv::Mat colorized = colorize(mat, colormap, maximize_contrast);

		if(!window_name.empty())
		{
			cv::imshow(window_name, colorized);
			cv::waitKey(1);
		}

		return colorized;
	}

	// -----------------------------------------------------------------------------------
	static cv::Mat unary2mat(const Eigen::VectorXd &unary, const size_t height,
							 const size_t width, const size_t label_id,
							 const int colormap = cv::COLORMAP_JET,
							 const std::string &window_name = std::string(),
							 const bool maximize_contrast = false)
	// -----------------------------------------------------------------------------------
	{
		const size_t n_classes = unary.size() / (width * height);

		if(n_classes < label_id)
			throw std::invalid_argument("VisualizationUtils::unary2mat - n_classes < label_id");

		cv::Mat mat(static_cast<int>(height), static_cast<int>(width), CV_64FC1);

		#pragma omp parallel for
		for(auto y = 0; y < height; ++y)
			for(auto x = 0; x < width; ++x)
				mat.at<double>(y, x) = unary[(y * width + x) * n_classes + label_id];

		const cv::Mat colorized = colorize(mat, colormap, maximize_contrast);

		if(!window_name.empty())
		{
			cv::imshow(window_name, colorized);
			cv::waitKey(1);
		}

		return colorized;
	}

	// -----------------------------------------------------------------------------------
	template <typename T>
	static std::vector<T> indicators2MAP(const std::vector<T> &indicators, const size_t n_classes)
	// -----------------------------------------------------------------------------------
	{
		const size_t n_pixels = indicators.size() / n_classes;
		
		std::vector<T> res(n_pixels, 0);

		for(size_t i = 0; i < n_pixels; ++i)
		{
			T maximum = indicators[(i * n_classes) + 0];
			res[i] = 0;

			for(size_t j = 1; j < n_classes; ++j)
				if(indicators[(i * n_classes) + j] > maximum)
				{
					maximum = indicators[(i * n_classes) + j];
					res[i] = static_cast<T>(j);
				}
		}

		return res;
	}

	// -----------------------------------------------------------------------------------
	template <typename T>
	static std::vector<T> indicators2MAP(const Eigen::VectorXd &indicators, const size_t n_classes)
	// -----------------------------------------------------------------------------------
	{
		const size_t n_pixels = indicators.size() / n_classes;

		std::vector<T> res(n_pixels, 0);

		for(size_t i = 0; i < n_pixels; ++i)
		{
			T maximum = static_cast<T>(indicators[(i * n_classes) + 0]);
			res[i] = 0;

			for(size_t j = 1; j < n_classes; ++j)
			if(indicators[(i * n_classes) + j] > maximum)
			{
				maximum = static_cast<T>(indicators[(i * n_classes) + j]);
				res[i] = static_cast<T>(j);
			}
		}

		return res;
	}

	// -----------------------------------------------------------------------------------
	static std::vector<cv::Mat> solutionIndicators2mats(const cv::Mat &solution, const size_t height,
														const size_t width)
	// -----------------------------------------------------------------------------------
	{
		const size_t n_classes = solution.cols / (height * width);

		std::vector<cv::Mat> res(solution.rows, cv::Mat());

		for(size_t i = 0; i < solution.rows; ++i)
		{
			const std::vector<double> current_indicators = OpenCVUtils::matRow2vct<double>(solution, i);
			const std::vector<double> current_map = indicators2MAP<double>(current_indicators, n_classes);
			const cv::Mat map = OpenCVUtils::vct2mat(current_map, height);
			//const cv::Mat colorized = colorize(map, cv::COLORMAP_HOT);
			res[i] = map.clone();
		}

		return res;
	}

	// -----------------------------------------------------------------------------------
	static std::vector<cv::Mat> solutionIndicators2mats(const Eigen::MatrixXd &mat,
														const size_t height,
														const size_t width)
	// -----------------------------------------------------------------------------------
	{
		const cv::Mat k_solutions = Eigen2OpenCVUtils::eigen2opencv(mat);
		const std::vector<cv::Mat> res = VisualizationUtils::solutionIndicators2mats(k_solutions, height, width);

		return res;
	}


	// -----------------------------------------------------------------------------------
	static std::vector<unsigned short> double2labels(const cv::Mat &doubles)
	// -----------------------------------------------------------------------------------
	{
		//std::vector<unsigned short> res(doubles.rows * doubles.cols);

		cv::Mat ushort_mat;
		doubles.convertTo(ushort_mat, CV_8UC1);
		std::vector<unsigned short> res(ushort_mat.data, ushort_mat.data + (ushort_mat.cols * ushort_mat.rows));//ushort_mat.dataend);

		return res;
	}


	// -----------------------------------------------------------------------------------
	template <typename T>
	static cv::Mat plotGraph(std::vector<T>& vals, int YRange[2])
	// -----------------------------------------------------------------------------------
	{

		auto it = std::minmax_element(vals.begin(), vals.end());
		float scale = 1. / std::ceil(*it.second - *it.first);
		float bias = *it.first;
		int rows = YRange[1] - YRange[0] + 1;
		cv::Mat image = cv::Mat::zeros(rows, vals.size(), CV_8UC3);
		image.setTo(0);
		for(int i = 0; i < static_cast<int>(vals.size()) - 1; i++)
		{
			cv::line(image, cv::Point(i, rows - 1 - (vals[i] - bias)*scale*YRange[1]),
					 cv::Point(i + 1, rows - 1 - (vals[i + 1] - bias)*scale*YRange[1]), cv::Scalar(255, 0, 0), 1);
		}

		return image;
	}

};

