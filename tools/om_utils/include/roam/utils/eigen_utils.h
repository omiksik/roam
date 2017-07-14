#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::VectorXi)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::VectorXd)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix2d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::MatrixXd)

#include <fstream>
#include <sdpp/configuration.h>
#include <sdpp/utils/math_utils.h>

// -----------------------------------------------------------------------------------
namespace ROAM
// -----------------------------------------------------------------------------------
{
	class EigenUtils
	{
	public:
		// -----------------------------------------------------------------------------------
		static Eigen::MatrixXd appendRow(const Eigen::MatrixXd &mat, const Eigen::RowVectorXd &vct)
		// -----------------------------------------------------------------------------------
		{
			assert(mat.cols() == vct.size());

			Eigen::MatrixXd res = mat;
			res.conservativeResize(res.rows() + 1, res.cols());
			res.row(res.rows() - 1) = vct;

			return res;
		}

		// -----------------------------------------------------------------------------------
		static Eigen::MatrixXd removeRow(const Eigen::MatrixXd& mat, unsigned int rowToRemove)
		// -----------------------------------------------------------------------------------
		{
			if(mat.rows() == 0)
				return mat;

			auto numRows = mat.rows() - 1;
			auto numCols = mat.cols();

			Eigen::MatrixXd res = mat;

			if(rowToRemove < numRows)
				res.block(rowToRemove, 0, numRows - rowToRemove, numCols) = mat.block(rowToRemove + 1, 0, numRows - rowToRemove, numCols);

			res.conservativeResize(numRows, numCols);

			return res;
		}

		// -----------------------------------------------------------------------------------
		static Eigen::MatrixXd removeColumn(const Eigen::MatrixXd& mat, unsigned int colToRemove)
		// -----------------------------------------------------------------------------------
		{
			if(mat.cols() == 0)
				return mat;

			auto numRows = mat.rows();
			auto numCols = mat.cols() - 1;

			Eigen::MatrixXd res = mat;

			if(colToRemove < numCols)
				res.block(0, colToRemove, numRows, numCols - colToRemove) = mat.block(0, colToRemove + 1, numRows, numCols - colToRemove);

			res.conservativeResize(numRows, numCols);

			return res;
		}

		/// expands N x N matrix into (N + 1) x (N + 1)
		/// elements [1:N, (N + 1)] = vct
		/// elements [(N + 1), 1:N] = vct^T
		/// element  [(N + 1), (N + 1)] = 1	
		// -----------------------------------------------------------------------------------
		static Eigen::MatrixXd addSymmetricRowColumn(const Eigen::MatrixXd &mat, const Eigen::VectorXd &vct)
		// -----------------------------------------------------------------------------------
		{
			if(mat.cols() != mat.rows())
				throw std::invalid_argument("EigenUtils::addSymmetricRowColumn - mat must be symmetric");

			if(mat.cols() != (vct.size()))
				throw std::invalid_argument("EigenUtils::addSymmetricRowColumn - mat.cols & vct dimensions must agree");

			// create full mat
			Eigen::MatrixXd res = Eigen::MatrixXd::Ones(mat.rows() + 1, mat.rows() + 1);

			// place mat 
			res.block(0, 0, res.rows() - 1, res.cols() - 1) = mat;

			// insert row
			res.block(res.rows() - 1, 0, 1, res.cols() - 1) = vct.transpose();

			// insert col
			res.block(0, res.cols() - 1, res.cols() - 1, 1) = vct; // .transpose();

			return res;
		}

		/// is_inf_or_nan(x) is just isnan(x-x)
		// -----------------------------------------------------------------------------------
		template<typename Derived>
		static inline bool is_finite(const Eigen::MatrixBase<Derived>& x)
		// -----------------------------------------------------------------------------------
		{
			return ((x - x).array() == (x - x).array()).all();
		}

		/// isnan(x) is just x==x
		// -----------------------------------------------------------------------------------
		template<typename Derived>
		static inline bool is_nan(const Eigen::MatrixBase<Derived>& x)
		// -----------------------------------------------------------------------------------
		{
			return ((x.array() == x.array())).all();
		}

		
		// -----------------------------------------------------------------------------------
		template<typename Derived>
		static inline bool almost_equal(const Eigen::MatrixBase<Derived> &x,
										const Eigen::MatrixBase<Derived> &y,
										int ulp = 2)
		// -----------------------------------------------------------------------------------
		{
			assert(x.size() == y.size());

			bool res = true;

			for(auto i = 0; i < x.size(); ++i)
				if(!MathUtils::almost_equal(x[i], y[i], ulp))
					res = false;

			return res;
		}

		// -----------------------------------------------------------------------------------
		template<typename Derived>
		static inline bool elements_near(const Eigen::MatrixBase<Derived> &x,
										 const Eigen::MatrixBase<Derived> &y,
										 const double abs_eps)
		// -----------------------------------------------------------------------------------
		{
			assert(x.size() == y.size());

			bool res = true;

			for(auto i = 0; i < x.size(); ++i)
				if((x(i) - y(i)) > abs_eps)
					res = false;

			return res;
		}

		// -----------------------------------------------------------------------------------
		template<typename Derived>
		static inline bool elements_near_rel(const Eigen::MatrixBase<Derived> &x,
											 const Eigen::MatrixBase<Derived> &y,
											 const double rel_eps)
		// -----------------------------------------------------------------------------------
		{
			assert(x.size() == y.size());

			bool res = true;

			for(auto i = 0; i < x.size(); ++i)
				if(!MathUtils::relative_eps_convergence(x[i], y[i], rel_eps))
					res = false;

			return res;
		}

		/// save eigen matrix
		// -----------------------------------------------------------------------------------
		template<class Matrix>
		static void write_binary(const std::string &filename, const Matrix& matrix)
		// -----------------------------------------------------------------------------------
		{
			std::ofstream out(filename, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);

			const typename Matrix::Index n_rows = matrix.rows();
			const typename Matrix::Index n_cols = matrix.cols();
				
			out.write((char*)(&n_rows), sizeof(typename Matrix::Index));
			out.write((char*)(&n_cols), sizeof(typename Matrix::Index));
			out.write((char*)(matrix.data()), n_rows * n_cols * sizeof(typename Matrix::Scalar));

			out.close();
		}

		/// load eigen matrix
		// -----------------------------------------------------------------------------------
		template<class Matrix>
		static void read_binary(const std::string &filename, Matrix& matrix)
		// -----------------------------------------------------------------------------------
		{
			std::ifstream in(filename, std::ofstream::in | std::ofstream::binary);
			
			typename Matrix::Index n_rows = matrix.rows();
			typename Matrix::Index n_cols = matrix.cols();

			in.read((char *)(&n_rows), sizeof(typename Matrix::Index));
			in.read((char *)(&n_cols), sizeof(typename Matrix::Index));
			
			matrix.resize(n_rows, n_cols);

			in.read((char *)matrix.data(), n_rows * n_cols * sizeof(typename Matrix::Scalar));
			
			in.close();
		}

		/// load eigen matrix
		// -----------------------------------------------------------------------------------
		template<class Matrix>
		static Matrix read_binary(const std::string &filename)
		// -----------------------------------------------------------------------------------
		{
			Matrix matrix;
			read_binary(filename, matrix);

			return matrix;
		}
	};
}