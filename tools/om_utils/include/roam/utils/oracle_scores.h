#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>

#include <sdpp/utils/confusion_matrix.h>

// TODO: remove
#include "opencv_utils.h"

// -----------------------------------------------------------------------------------
namespace ROAM
// -----------------------------------------------------------------------------------
{
	// -----------------------------------------------------------------------------------
	class OracleScores
	// -----------------------------------------------------------------------------------
	{
	public:
		// -----------------------------------------------------------------------------------
		explicit OracleScores(/*const size_t n_labels,*/ const size_t k,
							  const size_t n_expected_images = 1000)
							  : m_K(k)//, m_nLabels(n_labels)
		// -----------------------------------------------------------------------------------
		{
			Reset(n_expected_images);
		}

		// -----------------------------------------------------------------------------------
		virtual ~OracleScores()
		// -----------------------------------------------------------------------------------
		{
		}

		// -----------------------------------------------------------------------------------
		void Reset(const size_t n_expected_images = 1000)
		// -----------------------------------------------------------------------------------
		{
			m_oracle_scores.clear();
			m_oracle_matrices.clear();

			m_oracle_scores.reserve(n_expected_images);
			m_oracle_matrices.reserve(n_expected_images);
		}

		// -----------------------------------------------------------------------------------
		void Accumulate(const std::vector<int> &gt,
						const Eigen::MatrixXd &k_solutions,
						const size_t n_labels,
						const std::vector<double> &weights = std::vector<double>())
		// -----------------------------------------------------------------------------------
		{
			//assert(k_solutions.rows() == m_K);
			assert(static_cast<size_t>(k_solutions.rows()) <= m_K);
			Eigen::MatrixXd tmp_solutions = k_solutions;
			if(static_cast<size_t>(k_solutions.rows()) < m_K)
			{
				tmp_solutions = Eigen::MatrixXd(m_K, gt.size() * n_labels);

				for(auto i = 0; i < k_solutions.rows(); ++i)
					tmp_solutions.row(i) = k_solutions.row(i);

				// do we need this?
				for(size_t i = k_solutions.rows(); i < m_K; ++i)
					tmp_solutions.row(i) = k_solutions.row(k_solutions.rows() - 1);
			}

			// pre-compute confusion matrices
			const std::vector<ConfusionMatrix> matrices = confusionMatrices(gt, tmp_solutions, n_labels, weights);

			// store the data
			std::vector<double> c_oracle_scores(m_K);
			std::vector<ConfusionMatrix> c_oracle_matrices(m_K, ConfusionMatrix(n_labels));

			// compute oracle scores for 1, 2, ... K solutions
			#pragma omp parallel for
			for(auto i = 0; i < m_K; ++i)
			{
				// use oracle for the first 1, ... i matrices
				const std::vector<ConfusionMatrix> restricted(matrices.begin(), matrices.begin() + i + 1);

				const size_t best_solution = oracle(restricted);
				const ConfusionMatrix &oracle_matrix = matrices[best_solution];

				c_oracle_scores[i] = oracleScore(oracle_matrix);
				c_oracle_matrices[i] = oracle_matrix;
			}

			m_oracle_scores.push_back(c_oracle_scores);
			m_oracle_matrices.push_back(c_oracle_matrices);
		}

		// -----------------------------------------------------------------------------------
		double GlobalScore() const
		// -----------------------------------------------------------------------------------
		{
			return KthGlobalScore(m_K - 1);
		}

		// -----------------------------------------------------------------------------------
		std::vector<double> GlobalScores() const
		// -----------------------------------------------------------------------------------
		{
			std::vector<double> scores(m_K);

			#pragma omp parallel for
			for(auto i = 0; i < m_K; ++i)
				scores[i] = KthGlobalScore(i);

			return scores;
		}

		// -----------------------------------------------------------------------------------
		double KthGlobalScore(const size_t k) const
		// -----------------------------------------------------------------------------------
		{
			assert(k < m_K);

			const std::vector<double> scores = KthOraclePerImageScores(k);
			const double accumulate = std::accumulate(scores.begin(), scores.end(), 0.0);
			const double score = accumulate / static_cast<double>(scores.size());

			return score;
		}

		// -----------------------------------------------------------------------------------
		size_t NAccumulatedImages() const
		// -----------------------------------------------------------------------------------
		{
			return m_oracle_scores.size();
		}

		// -----------------------------------------------------------------------------------
		std::vector<double> PerImageScores(const size_t i) const
		// -----------------------------------------------------------------------------------
		{
			assert(i < m_oracle_scores.size());
			return m_oracle_scores[i];
		}

		// -----------------------------------------------------------------------------------
		std::vector<ConfusionMatrix> PerImageConfusionMatrices(const size_t i) const
		// -----------------------------------------------------------------------------------
		{
			assert(i < m_oracle_matrices.size());
			return m_oracle_matrices[i];
		}

		// -----------------------------------------------------------------------------------
		std::vector<double> KthOraclePerImageScores(const size_t k) const
		// -----------------------------------------------------------------------------------
		{
			const size_t nImages = NAccumulatedImages();

			std::vector<double> res(nImages);
			if(nImages < 1)
				return res;

			assert(k < m_oracle_scores[0].size());

			#pragma omp parallel for
			for(auto i = 0; i < nImages; ++i)
				res[i] = m_oracle_scores[i][k];

			return res;
		}

		// -----------------------------------------------------------------------------------
		std::vector<ConfusionMatrix> KthOraclePerImageConfusionMatrices(const size_t k) const
		// -----------------------------------------------------------------------------------
		{
			const size_t nImages = NAccumulatedImages();

			std::vector<ConfusionMatrix> res(nImages, ConfusionMatrix(0));
			if(nImages < 1)
				return res;

			assert(k < m_oracle_matrices[0].size());

			#pragma omp parallel for
			for(auto i = 0; i < nImages; ++i)
				res[i] = m_oracle_matrices[i][k];

			return res;
		}

		// -----------------------------------------------------------------------------------
		void Print(std::ostream &out = std::cout, const size_t decimal_prec = 4) const
		// -----------------------------------------------------------------------------------
		{
			// -----------------------------------------------------------------------------------
			// set precision
			out << std::setprecision(decimal_prec) << std::fixed;

			// -----------------------------------------------------------------------------------
			// cumulative scores
			const std::vector<double> scores = GlobalScores();

			out << "Cumulative scores (k = 1, k = 2, ..., k = K)" << std::endl;
			out << "k = ";
			for(size_t i = 0; i < m_K; ++i)
				out << "\t" << i;

			out << std::endl << "score: ";
			for(size_t i = 0; i < m_K; ++i)
				out << "\t" << scores[i];
			out << std::endl << std::endl;

			// -----------------------------------------------------------------------------------
			// score 
			out << "Oracle Global Score for (K = " << m_K << "): "
				<< GlobalScore() << std::scientific << std::endl;
		}

		// -----------------------------------------------------------------------------------
		friend std::ostream& operator<<(std::ostream& stream, const OracleScores& scores)
		// -----------------------------------------------------------------------------------
		{
			scores.Print(stream);
			return stream;
		}

	protected:
		// TODO: can be overloaded for any other performance measure
		// -----------------------------------------------------------------------------------
		virtual double oracleScore(const ConfusionMatrix &confusion_matrix) const
		// -----------------------------------------------------------------------------------
		{
			return confusion_matrix.Accuracy();
		}

		// -----------------------------------------------------------------------------------
		size_t oracle(const std::vector<ConfusionMatrix> &matrices) const
		// -----------------------------------------------------------------------------------
		{
			size_t best_id = 0;
			double best_score = 0.0;

			for(size_t i = 0; i < matrices.size(); ++i)
			{
				const ConfusionMatrix &cmat = matrices[i];
				const double score = oracleScore(cmat);
				if(best_score < score)
				{
					best_score = score;
					best_id = i;
				}
			}

			return best_id;
		}

		// -----------------------------------------------------------------------------------
		std::vector<ConfusionMatrix> confusionMatrices(const std::vector<int> &gt,
													   const Eigen::MatrixXd &k_solutions,
													   const size_t n_labels,
													   const std::vector<double> &weights = std::vector<double>()) const
		// -----------------------------------------------------------------------------------
		{
			assert(k_solutions.rows() == m_K);

			std::vector<ConfusionMatrix> matrices(m_K, ConfusionMatrix(n_labels));

			#pragma omp parallel for
			for(auto i = 0; i < m_K; ++i)
			{
				const Eigen::VectorXd &solution = k_solutions.row(i);
				const std::vector<int> predictions = VisualizationUtils::indicators2MAP<int>(solution, n_labels);
				assert(gt.size() == predictions.size());

				ConfusionMatrix cmat(n_labels);
				cmat.Accumulate(gt, predictions, weights);

				matrices[i] = cmat;
			}

			return matrices;
		}

		std::vector<std::vector<double> > m_oracle_scores;
		std::vector<std::vector<ConfusionMatrix> > m_oracle_matrices;

		const size_t m_K;
		//const size_t m_nLabels;
	};
}