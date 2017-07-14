#pragma once
#include <vector>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <set>

#include "../../../../../roam/include/Configuration.h"

// -----------------------------------------------------------------------------------
namespace ROAM
// -----------------------------------------------------------------------------------
{
	/// \brief Utility class providing confusion matrix, micro, macro scores, etc
	///
	///  Provides confusion matrix and various metrics
	// -----------------------------------------------------------------------------------
	class ConfusionMatrix
	// -----------------------------------------------------------------------------------
	{
	public:
		// -----------------------------------------------------------------------------------
		explicit ConfusionMatrix(const size_t nbr_labels) : m_nLabels(nbr_labels)
		// -----------------------------------------------------------------------------------
		{
			Reset();
		}

		// -----------------------------------------------------------------------------------
		void Reset()
		// -----------------------------------------------------------------------------------
		{
			m_cmat.assign(m_nLabels, std::vector<double>(m_nLabels, 0.0));
		}

		// -----------------------------------------------------------------------------------
		void Accumulate(const int gt,
						const int predicted,
						const double weight = 1.0)
		// -----------------------------------------------------------------------------------
		{
			if((gt < 0) || (predicted < 0))
			{
				std::cout << "ConfusionMatrix::Accumulate - label < 0" << std::endl;
				return;
			}

			m_cmat[gt][predicted] += weight;
		}

		// -----------------------------------------------------------------------------------
		void Accumulate(const std::vector<int> &gt,
						const std::vector<int> &predicted,
						const std::vector<double> &weights = std::vector<double>())
		// -----------------------------------------------------------------------------------
		{
			const size_t n = gt.size();
			const std::vector<double> w = (weights.size() == 0) ? std::vector<double>(n, 1.0) : weights;

			if((n != predicted.size()) || (n != w.size())) 
				throw std::invalid_argument("ConfusionMatrix::Accumulate - dimensions must agree");

			for(size_t i = 0; i < n; ++i)
			{
				if((gt[i]) < 0 || (predicted[i] < 0))
					continue;

				assert(gt[i] < m_nLabels);
				assert(predicted[i] < m_nLabels);

				Accumulate(gt[i], predicted[i], w[i]);
			}
		}

		// -----------------------------------------------------------------------------------
		void Accumulate(const ConfusionMatrix &other)
		// -----------------------------------------------------------------------------------
		{
			assert(other.NLabels() == m_nLabels);
			const std::vector<std::vector<double> > other_mat = other.GetMatrix();

			for(size_t l1 = 0; l1 < m_nLabels; ++l1)
				for(size_t l2 = 0; l2 < m_nLabels; ++l2)
					m_cmat[l1][l2] += other_mat[l1][l2];
		}

		/// precision := TP / (TP + FP); for a particular label
		/// fraction of events correctly declared $i$ out of all instances where the algorithm declared $i$
		// -----------------------------------------------------------------------------------
		double Precision(const size_t label) const
		// -----------------------------------------------------------------------------------
		{
			if(label >= m_nLabels)
				throw std::invalid_argument("ConfusionMatrix::Precision - label must be < n_labels");

			double col_sum = 0.0;
			for(auto l = 0; l < m_nLabels; l++)
				col_sum += m_cmat[l][label];

			// well, it depends, if union_size == 0, it's perfect overlap...
			return (col_sum == 0) ? 0.0 : (m_cmat[label][label] / col_sum);
		}

		/// precision := TP / (TP + FP);
		// -----------------------------------------------------------------------------------
		std::vector<double> Precision() const
		// -----------------------------------------------------------------------------------
		{
			std::vector<double> vct(m_nLabels, 0.0);
			#pragma omp parallel for
			for(auto l = 0; l < m_nLabels; l++)
				vct[l] = Precision(l);

			return vct;
		}

		/// precision := TP / (TP + FP); weighted by class fractions
		// -----------------------------------------------------------------------------------
		double MicroPrecision() const
		// -----------------------------------------------------------------------------------
		{
			const std::vector<double> vct_p = Precision();
			const std::vector<double> vct_fracs = ComputeLabelFractions();
			const double micro_p = std::inner_product(vct_p.begin(), vct_p.end(), vct_fracs.begin(), 0.0);

			return micro_p;
		}

		/// averaged precision
		// -----------------------------------------------------------------------------------
		double MacroPrecision() const
		// -----------------------------------------------------------------------------------
		{
			const std::vector<double> vct_p = Precision();
			const size_t n_active_cls = NActiveLabels();

			return (n_active_cls == 0) ? 0 : std::accumulate(vct_p.begin(), vct_p.end(), 0.0) / static_cast<double>(n_active_cls);
		}

		/// recall := TP / (TP + FN)
		/// fraction of events correctly declared $i$ out of all of the cases where the true of state is $i$
		// -----------------------------------------------------------------------------------
		double Recall(const size_t label) const
		// -----------------------------------------------------------------------------------
		{
			if(label >= m_nLabels)
				throw std::invalid_argument("ConfusionMatrix::Recall - label must be < n_labels");

			const double row_sum = std::accumulate(m_cmat[label].begin(), m_cmat[label].end(), 0.0);

			if(row_sum == 0)
				return 0.0;

			return m_cmat[label][label] / row_sum;
		}

		/// recall := TP / (TP + FN)
		// -----------------------------------------------------------------------------------
		std::vector<double> Recall() const
		// -----------------------------------------------------------------------------------
		{
			std::vector<double> vct(m_nLabels, 0.0);
			#pragma omp parallel for
			for(auto l = 0; l < m_nLabels; l++)
				vct[l] = Recall(l);

			return vct;
		}

		/// recall := TP / (TP + FN); weighted by class fractions
		// -----------------------------------------------------------------------------------
		double MicroRecall() const
		// -----------------------------------------------------------------------------------
		{
			const std::vector<double> vct_r = Recall();
			const std::vector<double> vct_fracs = ComputeLabelFractions();
			const double micro_r = std::inner_product(vct_r.begin(), vct_r.end(), vct_fracs.begin(), 0.0);

			return micro_r;
		}

		/// averaged recall
		// -----------------------------------------------------------------------------------
		double MacroRecall() const
		// -----------------------------------------------------------------------------------
		{
			const std::vector<double> vct_r = Recall();
			const size_t n_active_cls = NActiveLabels();

			return (n_active_cls == 0) ? 0 : std::accumulate(vct_r.begin(), vct_r.end(), 0.0) / static_cast<double>(n_active_cls);
		}

		// -----------------------------------------------------------------------------------
		double FBeta(const size_t label, const double beta = 1.0) const
		// -----------------------------------------------------------------------------------
		{
			if(label >= m_nLabels)
				throw std::invalid_argument("ConfusionMatrix::FBeta - label must be < n_labels");

			const double r = Recall(label);
			const double p = Precision(label);

			if(!std::isfinite(p) || !std::isfinite(r))
				return 0.0;

			const double beta2 = beta * beta;
			const double num = (1.0 + beta2) * p * r;
			const double den = ((beta2 * p) + r);

			return (den == 0) ? 0.0 : (num / den);
		}

		// -----------------------------------------------------------------------------------
		std::vector<double> FBeta(const double beta = 1.0) const
		// -----------------------------------------------------------------------------------
		{
			std::vector<double> vct(m_nLabels, 0.0);
			#pragma omp parallel for
			for(auto l = 0; l < m_nLabels; l++)
				vct[l] = FBeta(l, beta);

			return vct;
		}

		// -----------------------------------------------------------------------------------
		double MicroFBeta(const double beta = 1.0) const
		// -----------------------------------------------------------------------------------
		{
			const std::vector<double> vct_fbeta = FBeta(beta);
			const std::vector<double> vct_fracs = ComputeLabelFractions();
			const double micro_fbeta = std::inner_product(vct_fbeta.begin(), vct_fbeta.end(), vct_fracs.begin(), 0.0);

			return micro_fbeta;
		}

		// -----------------------------------------------------------------------------------
		double MacroFBeta(const double beta = 1.0) const
		// -----------------------------------------------------------------------------------
		{
			const std::vector<double> vct_fbeta = FBeta(beta);
			const size_t n_active_cls = NActiveLabels();

			return (n_active_cls == 0) ? 0 : std::accumulate(vct_fbeta.begin(), vct_fbeta.end(), 0.0) / static_cast<double>(n_active_cls);
		}

		// -----------------------------------------------------------------------------------
		double F1(const size_t label) const
		// -----------------------------------------------------------------------------------
		{
			return FBeta(label, 1.0);
		}

		// -----------------------------------------------------------------------------------
		std::vector<double> F1() const
		// -----------------------------------------------------------------------------------
		{
			return FBeta(1.0);
		}

		// -----------------------------------------------------------------------------------
		double MicroF1() const
		// -----------------------------------------------------------------------------------
		{
			return MicroFBeta(1.0);
		}

		// -----------------------------------------------------------------------------------
		double MacroF1() const
		// -----------------------------------------------------------------------------------
		{
			return MacroFBeta(1.0);
		}

		/// Computes accuracy := trace(conf_mat) / sum(conf_mat)
		// -----------------------------------------------------------------------------------
		double Accuracy() const
		// -----------------------------------------------------------------------------------
		{
			double trace = 0.0;
			double total = 0.0;
			for(size_t i = 0; i < m_nLabels; i++)
			{
				trace += m_cmat[i][i];
				total += std::accumulate(m_cmat[i].begin(), m_cmat[i].end(), 0.0);
			}
			return trace / total;
		}

		/// Jacard index := TP / (TP + FP + FN); N_{ii} / (-N_{ii} + \sum_j (N_{ij} + N_{ji}))
		// -----------------------------------------------------------------------------------
		double IoU(const size_t label) const
		// -----------------------------------------------------------------------------------
		{
			if(label >= m_nLabels)
				throw std::invalid_argument("ConfusionMatrix::IoU - label must be < n_labels");

			const double intersection_size = m_cmat[label][label];

			double col_sum = 0.0;
			for(auto l = 0; l < m_nLabels; l++)
				col_sum += m_cmat[l][label];

			const double row_sum = std::accumulate(m_cmat[label].begin(), m_cmat[label].end(), 0.0);

			const double union_size = row_sum + col_sum - m_cmat[label][label];

			// well, it depends, if union_size == 0, it's perfect overlap...
			return (intersection_size == 0) ? 0.0 : intersection_size / union_size;
		}

		/// Jacard index := TP / (TP + FP + FN); N_{ii} / (-N_{ii} + \sum_j (N_{ij} + N_{ji}))
		// -----------------------------------------------------------------------------------
		std::vector<double> IoU() const
		// -----------------------------------------------------------------------------------
		{
			std::vector<double> vct(m_nLabels, 0.0);
			#pragma omp parallel for
			for(auto l = 0; l < m_nLabels; l++)
				vct[l] = IoU(l);

			return vct;
		}

		// -----------------------------------------------------------------------------------
		double MicroIoU() const
		// -----------------------------------------------------------------------------------
		{
			const std::vector<double> vct_iou = IoU();
			const std::vector<double> vct_fracs = ComputeLabelFractions();
			const double micro_fbeta = std::inner_product(vct_iou.begin(), vct_iou.end(), vct_fracs.begin(), 0.0);

			return micro_fbeta;
		}

		// -----------------------------------------------------------------------------------
		double MacroIoU() const
		// -----------------------------------------------------------------------------------
		{
			const std::vector<double> vct_iou = IoU();
			const size_t n_active_cls = NActiveLabels();

			return (n_active_cls == 0) ? 0 : std::accumulate(vct_iou.begin(), vct_iou.end(), 0.0) / static_cast<double>(n_active_cls);
		}

		// -----------------------------------------------------------------------------------
		void PrintConfusionMatrix(std::ostream &out = std::cout) const
		// -----------------------------------------------------------------------------------
		{
			for(size_t i = 0; i < m_nLabels; i++)
			{
				for(size_t j = 0; j < m_nLabels; j++)
					out << m_cmat[i][j] << " ";

				out << std::endl;
			}
			out << std::endl;
		}

		// -----------------------------------------------------------------------------------
		void PrintRowNormalizedConfusionMatrix(std::ostream &out = std::cout) const
		// -----------------------------------------------------------------------------------
		{
			std::vector<std::vector<double> > row_normalized = GetRowNormalizedMatrix();

			for(size_t i = 0; i < m_nLabels; i++)
			{
				for(size_t j = 0; j < m_nLabels; j++)
					out << row_normalized[i][j] << " ";

				out << std::endl;
			}
			out << std::endl;
		}

		// -----------------------------------------------------------------------------------
		void PrintColNormalizedConfusionMatrix(std::ostream &out = std::cout) const
		// -----------------------------------------------------------------------------------
		{
			std::vector<std::vector<double> > col_normalized = GetColNormalizedMatrix();

			for(size_t i = 0; i < m_nLabels; i++)
			{
				for(size_t j = 0; j < m_nLabels; j++)
					out << col_normalized[i][j] << " ";

				out << std::endl;
			}
			out << std::endl;
		}

		// -----------------------------------------------------------------------------------
		void PrintLabelCounts(std::ostream &out = std::cout) const
		// -----------------------------------------------------------------------------------
		{
			const std::vector<double> vct_labels = ComputeLabelCounts();
			for(size_t k = 0; k < m_nLabels; k++)
				out << " " << vct_labels[k];

			out << std::endl;
		}

		// -----------------------------------------------------------------------------------
		void PrintPerClassPrecision(std::ostream &out = std::cout) const
		// -----------------------------------------------------------------------------------
		{
			const std::vector<double> vct_p = Precision();
			for(size_t k = 0; k < m_nLabels; k++)
				out << vct_p[k] << " ";

			out << std::endl;
		}

		// -----------------------------------------------------------------------------------
		void PrintPerClassRecall(std::ostream &out = std::cout) const
		// -----------------------------------------------------------------------------------
		{
			const std::vector<double> vct_r = Recall();
			for(size_t k = 0; k < m_nLabels; k++)
				out << vct_r[k] << " ";

			out << std::endl;
		}

		// -----------------------------------------------------------------------------------
		void PrintPerClassF1(std::ostream &out = std::cout) const
		// -----------------------------------------------------------------------------------
		{
			const std::vector<double> vct_f1 = F1();
			for(size_t k = 0; k < m_nLabels; k++)
				out << vct_f1[k] << " ";

			out << std::endl;
		}

		// -----------------------------------------------------------------------------------
		void PrintPerClassIoU(std::ostream &out = std::cout) const
		// -----------------------------------------------------------------------------------
		{
			const std::vector<double> vct_iou = IoU();
			for(size_t k = 0; k < m_nLabels; k++)
				out << vct_iou[k] << " ";

			out << std::endl;
		}

		// -----------------------------------------------------------------------------------
		void PrintPerClassStats(std::ostream &out = std::cout, const size_t decimal_prec = 4) const
		// -----------------------------------------------------------------------------------
		{
			// set precision
			out << std::setprecision(decimal_prec) << std::fixed;

			const std::vector<double> vct_p = Precision();
			const std::vector<double> vct_r = Recall();
			const std::vector<double> vct_f1 = F1();
			const std::vector<double> vct_iou = IoU();
			out << "Precision \t Recall \t F1 \t IoU" << std::endl;
			for(size_t k = 0; k < m_nLabels; k++)
				out << vct_p[k] << " \t " << vct_r[k] << " \t " << vct_f1[k] << " \t " << vct_iou[k] << std::endl;

			out << std::scientific << std::endl;
		}

		// -----------------------------------------------------------------------------------
		void PrintEmptyClasses(std::ostream &out = std::cout) const
		// -----------------------------------------------------------------------------------
		{
			const std::set<size_t> empty = GetEmptyClasses();

			for(auto it = empty.begin(); it != empty.end(); ++it)
				out << *it << " ";

			out << std::endl;
		}

		// -----------------------------------------------------------------------------------
		void Print(std::ostream &out = std::cout, const size_t decimal_prec = 4) const
		// -----------------------------------------------------------------------------------
		{
			// -----------------------------------------------------------------------------------
			// set precision
			out << std::setprecision(decimal_prec);

			// -----------------------------------------------------------------------------------
			// confusion matrix
			out << "Confusion matrix:" << std::endl;
			PrintConfusionMatrix(out);
			out << std::endl;

			// -----------------------------------------------------------------------------------
			// per class precision, recall, f1, IoU
			PrintPerClassStats(out);
			out << std::endl;

			// -----------------------------------------------------------------------------------
			// label counts
			out << "Label counts:";
			PrintLabelCounts(out);
			out << std::endl;

			// -----------------------------------------------------------------------------------
			// empty classes
			if(NActiveLabels() != m_nLabels)
			{
				out << "Empty labels: "; 
				PrintEmptyClasses(out);
				out << std::endl;
			}

			// -----------------------------------------------------------------------------------
			// micro precision, recall, f1
			out << std::fixed;
			out << "Micro Precision: " << MicroPrecision() << "   ";
			out << "Micro Recall: " << MicroRecall() << "   ";
			out << "Micro F1: " << MicroF1() << std::endl;

			// -----------------------------------------------------------------------------------
			// macro precision, recall, f1
			out << "Macro Precision: " << MacroPrecision() << "   ";
			out << "Macro Recall: " << MacroRecall() << "   ";
			out << "Macro F1: " << MacroF1() << std::endl;

			out << "Total accuracy: " << Accuracy() << std::scientific << std::endl;
		}

		// -----------------------------------------------------------------------------------
		friend std::ostream& operator<<(std::ostream& stream, const ConfusionMatrix& matrix)
		// -----------------------------------------------------------------------------------
		{
			matrix.Print(stream);
			return stream;
		}

		// -----------------------------------------------------------------------------------
		std::vector<double> ComputeLabelCounts() const
		// -----------------------------------------------------------------------------------
		{
			std::vector<double> vct_counts(m_nLabels, 0);
			#pragma omp parallel for
			for(auto l = 0; l < m_nLabels; l++)
				vct_counts[l] = std::accumulate(m_cmat[l].begin(), m_cmat[l].end(), 0.0);

			return vct_counts;
		}

		// -----------------------------------------------------------------------------------
		std::vector<double> ComputeLabelFractions() const
		// -----------------------------------------------------------------------------------
		{
			std::vector<double> vct_fracs = ComputeLabelCounts();
			const double total = std::accumulate(vct_fracs.begin(), vct_fracs.end(), 0.0);

			#pragma omp parallel for
			for(auto l = 0; l < m_nLabels; l++)
				vct_fracs[l] /= total;

			return vct_fracs;
		}

		/// Number of allocated classes
		// -----------------------------------------------------------------------------------
		size_t NLabels() const
		// -----------------------------------------------------------------------------------
		{
			return m_nLabels;
		}

		/// Number of classes represented present in confusion matrix
		// -----------------------------------------------------------------------------------
		size_t NActiveLabels() const
		// -----------------------------------------------------------------------------------
		{
			const std::vector<double> vct_counts = ComputeLabelCounts();
			return nActiveLabels(vct_counts);
		}

		/// Returns classes that are not represented in the confusion matrix
		// -----------------------------------------------------------------------------------
		std::set<size_t> GetEmptyClasses() const
		// -----------------------------------------------------------------------------------
		{
			const std::vector<double> vct_counts = ComputeLabelCounts();
			
			std::set<size_t> empty;
			for(size_t l = 0; l < m_nLabels; ++l)
				if(vct_counts[l] == 0)
					empty.insert(l);

			return empty;
		}

		// -----------------------------------------------------------------------------------
		std::vector<std::vector<double> > GetMatrix() const
		// -----------------------------------------------------------------------------------
		{
			return m_cmat;
		}

		// -----------------------------------------------------------------------------------
		std::vector<std::vector<double> > GetRowNormalizedMatrix() const
		// -----------------------------------------------------------------------------------
		{
			std::vector<std::vector<double> > row_normalized = GetMatrix();

			for(size_t l1 = 0; l1 < m_nLabels; ++l1)
			{
				const double row_sum = std::accumulate(row_normalized[l1].begin(), row_normalized[l1].end(), 0.0);
				for(size_t l2 = 0; l2 < m_nLabels; ++l2)
					row_normalized[l1][l2] = (row_sum == 0) ? 0.0 : row_normalized[l1][l2] / row_sum;
			}

			return row_normalized;
		}

		// -----------------------------------------------------------------------------------
		std::vector<std::vector<double> > GetColNormalizedMatrix() const
		// -----------------------------------------------------------------------------------
		{
			std::vector<std::vector<double> > col_normalized = GetMatrix();

			for(size_t l1 = 0; l1 < m_nLabels; ++l1)
			{
				double col_sum = 0.0;
				for(size_t l2 = 0; l2 < m_nLabels; ++l2)
					col_sum += col_normalized[l2][l1];

				for(size_t l2 = 0; l2 < m_nLabels; ++l2)
					col_normalized[l2][l1] = (col_sum == 0) ? 0.0 : col_normalized[l2][l1] / col_sum;
			}

			return col_normalized;
		}

	protected:
		/// Number of classes represented present in confusion matrix
		// -----------------------------------------------------------------------------------
		size_t nActiveLabels(const std::vector<double> &vct_counts) const
		// -----------------------------------------------------------------------------------
		{
			assert(vct_counts.size() == m_nLabels);

			size_t n_active_classes = 0;
			for(size_t l = 0; l < m_nLabels; ++l)
				if(vct_counts[l] > 0)
					n_active_classes++;

			return n_active_classes;
		}

		size_t m_nLabels;
		std::vector<std::vector<double> > m_cmat;
	};
}
