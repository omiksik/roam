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

#include <opencv2/opencv.hpp>
#include "../../../roam/include/Configuration.h"

#ifndef M_PI
    #define M_PI       3.14159265358979323846
#endif

namespace ROAM
{

/*!
* \brief GMM: a simple gmm model with online adaptation
*/
// -----------------------------------------------------------------------------------
struct GMMModel
// -----------------------------------------------------------------------------------
{
    // -----------------------------------------------------------------------------------
    struct MixtureComponent
    // -----------------------------------------------------------------------------------
    {
        MixtureComponent() : weight(0.0), mass(0) {}
        cv::Vec3f mean;			/// mean vector
        cv::Mat covariance;		/// covariance matrix
        cv::Mat iCovariance;	/// pre-computed inverted covariance matrix (save computation during Mahalanobis)
        float weight;			/// weight of a mixture
        size_t mass;			/// number of pixels from which the cluster is constructed
    };

    // -----------------------------------------------------------------------------------
    struct Parameters
    // -----------------------------------------------------------------------------------
    {
        // -----------------------------------------------------------------------------------
        explicit Parameters(const size_t k_ = 3)
        : attempts(5), k(k_), min_fraction(0.1f), use_binary_segment(true), update_k(3),
        update_min_fraction(0.1f), d_similar_merge(1.0f), d_similar_skip(1.5f), awu(10000), decay_factor(0.97f),
        discard_threshold(20), noise_factor(0.00005f)
        // -----------------------------------------------------------------------------------
        {
        }

        // -----------------------------------------------------------------------------------
        Parameters(const size_t n_frames1, const size_t n_frames2, const size_t patch_size)
            : Parameters(3)
        // -----------------------------------------------------------------------------------
        {
            const size_t n_pixels = (2 * patch_size) * (2 * patch_size);
            const size_t expected_per_cluster = static_cast<size_t>(n_pixels / static_cast<FLOAT_TYPE>(2 * k));

            awu = expected_per_cluster * n_frames1;
            decay_factor = static_cast<FLOAT_TYPE>(std::pow(awu, -1.0 / static_cast<FLOAT_TYPE>(n_frames2)));
            discard_threshold = std::max(static_cast<int>(expected_per_cluster * 0.01), 10);
        }

        size_t attempts;
        size_t k;
        FLOAT_TYPE min_fraction;

        bool use_binary_segment;

        size_t update_k;
        FLOAT_TYPE update_min_fraction;

        FLOAT_TYPE d_similar_merge;
        FLOAT_TYPE d_similar_skip;

        size_t awu;
        FLOAT_TYPE decay_factor;
        size_t discard_threshold;

        FLOAT_TYPE noise_factor;
    };

    // -----------------------------------------------------------------------------------
    explicit GMMModel(const Parameters &parameters = Parameters())
        : n_components(0), params(parameters), initialized(false)
    // -----------------------------------------------------------------------------------
    {
    }

    // -----------------------------------------------------------------------------------
    int initializeMixture(const cv::Mat &patch, const cv::Mat &mask)
    // -----------------------------------------------------------------------------------
    {
        gmm = getMixture(patch, mask, params.k, params.min_fraction);

        if(gmm.size() == 0)
            return -1;

        n_components = gmm.size();
        initialized = true;

        return 0;
    }

    // -----------------------------------------------------------------------------------
    int initializeMixture(const cv::Mat &patch, const cv::Mat &mask, const Parameters &parameters)
    // -----------------------------------------------------------------------------------
    {
        this->params = parameters;
        const int res = initializeMixture(patch, mask);

        return res;
    }

    // -----------------------------------------------------------------------------------
    int updateMixture(const cv::Mat &patch, const cv::Mat &mask,
        const std::vector<MixtureComponent> &other_gmm = std::vector<MixtureComponent>())
    // -----------------------------------------------------------------------------------
    {
        // approx corresponds to miksik et al., icra 2011

        // -----------------------------------------------------------------------------------
        // extract new clusters
        std::vector<MixtureComponent> new_clusters = getMixture(patch, mask, params.update_k, params.update_min_fraction);

        // -----------------------------------------------------------------------------------
        // prune new clusters => they won't contain clusters similar to any in other_gmm
        {
            const size_t n_components_other = other_gmm.size();
            std::vector<MixtureComponent>::iterator component = new_clusters.begin();
            while(component != new_clusters.end())
            {
                // throw away similar clusters
                bool deleted = false;
                for(size_t id = 0; id < n_components_other; ++id)
                {
                    const MixtureComponent &other_component = other_gmm[id];

                    const cv::Mat i_combined_covariance = (other_component.covariance + component->covariance).inv();

                    const FLOAT_TYPE similarity = static_cast<FLOAT_TYPE>(cv::Mahalanobis(other_component.mean, component->mean, i_combined_covariance));
                    const FLOAT_TYPE similarity_sq = std::pow(similarity, 2);

                    if(similarity_sq < params.d_similar_skip) // 1.5
                    {
                        component = new_clusters.erase(component);
                        deleted = true;
                        break;
                    }
                }

                if(deleted)
                    continue;
                else
                    component++;
            }
        }

        // -----------------------------------------------------------------------------------
        // update the most promissing 
        for(size_t id_new = 0; id_new < new_clusters.size(); ++id_new)
        {
            const MixtureComponent &new_mixture = new_clusters[id_new];

            FLOAT_TYPE min_distance = std::numeric_limits<FLOAT_TYPE>::infinity();
            size_t min_id = 0;

            // compute cluster similarity
            //
            // using n_components instead of "gmm.size()" is correct 
            // since we want overlaps only with the original clusters, not added in a current iteration
            for(size_t id_orig = 0; id_orig < n_components; ++id_orig)
            {
                const MixtureComponent &orig_mixture = gmm[id_orig];

                const cv::Mat i_combined_covariance = (orig_mixture.covariance + new_mixture.covariance).inv();

                const FLOAT_TYPE similarity = static_cast<FLOAT_TYPE>(cv::Mahalanobis(orig_mixture.mean, new_mixture.mean, i_combined_covariance));
                const FLOAT_TYPE similarity_sq = std::pow(similarity, 2);

                if(similarity_sq < min_distance)
                {
                    min_distance = similarity_sq;
                    min_id = id_orig;
                }
            }

            // do we have a similar cluster?
            if(min_distance < params.d_similar_merge) // 0.1
            {
                MixtureComponent &orig_mixture = gmm[min_id];

                const size_t normalizer = orig_mixture.mass + new_mixture.mass;

                orig_mixture.mean = (static_cast<float>(orig_mixture.mass) * orig_mixture.mean + static_cast<float>(new_mixture.mass) * new_mixture.mean) / static_cast<FLOAT_TYPE>(normalizer);
                orig_mixture.covariance = (static_cast<float>(orig_mixture.mass) * orig_mixture.covariance + static_cast<float>(new_mixture.mass) * new_mixture.covariance) / static_cast<FLOAT_TYPE>(normalizer);
                orig_mixture.iCovariance = orig_mixture.covariance.inv();
                orig_mixture.mass = normalizer;
            }
            else // we need to introduce a new component
            {
                gmm.push_back(new_mixture);
            }
        }

        // -----------------------------------------------------------------------------------
        // decay, awu and discarding
        size_t total_mass = 0;
        // const size_t n_components_other = other_gmm.size();
        std::vector<MixtureComponent>::iterator component = gmm.begin();
        while(component != gmm.end())
        {
            // anti-windup
            component->mass = std::min(component->mass, params.awu);

            // decay
            component->mass = static_cast<size_t>(component->mass * params.decay_factor);

            // discard small components
            if(component->mass < params.discard_threshold)
            {
                component = gmm.erase(component);
                continue;
            }

            // current mass to update gmm weights
            total_mass += component->mass;

            // and iterate
            component++;
        }

        // -----------------------------------------------------------------------------------
        // update w
        for(size_t id_orig = 0; id_orig < gmm.size(); ++id_orig)
            gmm[id_orig].weight = gmm[id_orig].mass / static_cast<FLOAT_TYPE>(total_mass);

        // update number of components
        n_components = gmm.size();

        if(gmm.size() == 0)
        {
            initialized = false;
            return -1;
        }

        initialized = true;

        return 0;
    }

    // -----------------------------------------------------------------------------------
    int updateMixture(const cv::Mat &patch, const cv::Mat &mask, const Parameters &parameters,
        const std::vector<MixtureComponent> &other_gmm = std::vector<MixtureComponent>())
    // -----------------------------------------------------------------------------------
    {
        this->params = parameters;
        return updateMixture(patch, mask, other_gmm);
    }

    // -----------------------------------------------------------------------------------
    FLOAT_TYPE getMahalanobisDistance(const cv::Vec3f &color) const
    // -----------------------------------------------------------------------------------
    {
        if(!initialized)
            throw std::logic_error("GMM needs to be initialized firts");

        float min_distance = std::numeric_limits<float>::infinity();

        for(size_t i = 0; i < n_components; ++i)
        {
            const MixtureComponent &component = gmm[i];

            const cv::Mat &inv = component.iCovariance;
            const FLOAT_TYPE distance = static_cast<FLOAT_TYPE>(cv::Mahalanobis(color, component.mean, inv));
            if(distance < min_distance)
                min_distance = distance;
        }

        return min_distance;
    }

    // -----------------------------------------------------------------------------------
    cv::Mat getMahalanobisDistance(const cv::Mat &patch) const
    // -----------------------------------------------------------------------------------
    {
        if(!initialized)
            throw std::logic_error("GMM needs to be initialized firts");

        cv::Mat output(patch.rows, patch.cols, CV_32FC1);

        cv::Mat patch_f;
        patch.convertTo(patch_f, CV_32FC1);

        for(int row = 0; row < patch.rows; ++row)
        for(int col = 0; col < patch.cols; ++col)
        {
            const cv::Vec3f &color = patch_f.at<cv::Vec3f>(row, col);
            const float distance = getMahalanobisDistance(color);

            output.at<float>(row, col) = distance;
        }

        return output;
    }

    /// evaluates gmm probability
    // -----------------------------------------------------------------------------------
    FLOAT_TYPE getLikelihood(const cv::Vec3f &color) const
    // -----------------------------------------------------------------------------------
    {
        if(!initialized)
            throw std::logic_error("GMM needs to be initialized first");

        FLOAT_TYPE probability = 0;

        for(size_t i = 0; i < n_components; ++i)
        {
            const MixtureComponent &component = gmm[i];

            const cv::Mat &cov = component.covariance;
            const cv::Mat &inv = component.iCovariance;
            const FLOAT_TYPE distance = static_cast<FLOAT_TYPE>(cv::Mahalanobis(color, component.mean, inv));

            const FLOAT_TYPE frac = static_cast<FLOAT_TYPE>(1.0 / (std::sqrt(std::pow(2 * M_PI, 3) * cv::determinant(cov))));
            probability += component.weight * frac * std::exp(-0.5f * std::pow(distance, 2));
        }

        return probability;
    }

    // -----------------------------------------------------------------------------------
    cv::Mat getLikelihood(const cv::Mat &patch) const
    // -----------------------------------------------------------------------------------
    {
        if(!initialized)
            throw std::logic_error("GMM needs to be initialized firts");

        cv::Mat output(patch.rows, patch.cols, CV_32FC1);
        output.setTo(0.0);

        cv::Mat patch_f;
        patch.convertTo(patch_f, CV_32FC1);

        std::vector<FLOAT_TYPE> fractions(gmm.size());
        for(size_t i = 0; i < n_components; ++i)
            fractions[i] = static_cast<FLOAT_TYPE>(gmm[i].weight * 1.0 / (std::sqrt(std::pow(2 * M_PI, 3) * cv::determinant(gmm[i].covariance))));

        for(int row = 0; row < patch.rows; ++row)
        {
            const cv::Vec3f* patch_row_ptr = patch_f.ptr<cv::Vec3f>(row);
            FLOAT_TYPE* outpt_row_ptr = output.ptr<FLOAT_TYPE>(row);

            for(int col = 0; col < patch.cols; ++col)
            {
                const cv::Vec3f &color = patch_row_ptr[col];
                const FLOAT_TYPE probability = getLikelihood(color, fractions);

                outpt_row_ptr[col] = probability;
            }
        }

        return output;
    }

    // -----------------------------------------------------------------------------------
    const std::vector<MixtureComponent> & getGMMComponents() const
    // -----------------------------------------------------------------------------------
    {
        return gmm;
    }

protected:
    /// evaluates gmm likelihood with precomputed fractional parts
    // -----------------------------------------------------------------------------------
    FLOAT_TYPE getLikelihood(const cv::Vec3f &color, std::vector<FLOAT_TYPE> &fractions) const
    // -----------------------------------------------------------------------------------
    {
        if(!initialized)
            throw std::logic_error("GMM needs to be initialized firts");

        FLOAT_TYPE probability = 0;

        for(size_t i = 0; i < n_components; ++i)
        {
            const MixtureComponent &component = gmm[i];

            // const cv::Mat &cov = component.covariance;
            const cv::Mat &inv = component.iCovariance;
            const FLOAT_TYPE distance = static_cast<FLOAT_TYPE>(cv::Mahalanobis(color, component.mean, inv));

            const FLOAT_TYPE &frac = fractions[i];
            probability += frac * std::exp(-0.5f * std::pow(distance, 2));
        }

        return probability;
    }

    // -----------------------------------------------------------------------------------
    std::vector<MixtureComponent> getMixture(const cv::Mat &patch, const cv::Mat &mask,
        const size_t k = 3, const FLOAT_TYPE minFraction = 0.1) const
    // -----------------------------------------------------------------------------------
    {
        assert(mask.type() == CV_8UC1);

        std::vector<MixtureComponent> mixture;

        const size_t &n_clusters = k;
        mixture.reserve(n_clusters);

        cv::Mat patch_f;
        patch.convertTo(patch_f, CV_32FC1);

        std::vector<cv::Vec3f> tmp;
        tmp.reserve(patch.total());

        for(int y = 0; y < patch.rows; ++y)
        for(int x = 0; x < patch.cols; ++x)
        if(mask.at<uchar>(y, x) != 0) // uchar
        {
            const cv::Vec3f &sample = patch_f.at<cv::Vec3f>(y, x);
            tmp.push_back(sample);
        }

        if(tmp.size() <= n_clusters)
            return mixture;

        cv::Mat labels, centers;
        const cv::Mat samples = cv::Mat(tmp).reshape(1);
        const float n_pixels = static_cast<float>(tmp.size());

        cv::kmeans(samples, static_cast<int>(n_clusters), labels, cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001),
            static_cast<int>(params.attempts), cv::KMEANS_PP_CENTERS, centers);

        size_t n_used_pixels = 0;

        // extract covariance matrices and centers
        for(size_t id = 0; id < n_clusters; ++id)
        {
            std::vector<cv::Vec3f> pts;
            pts.reserve(labels.rows);

            for(int pt = 0; pt < labels.rows; ++pt)
            if(labels.at<int>(pt, 0) == id)
            {
                const cv::Vec3f &sample = samples.row(pt);
                pts.push_back(sample);
            }

            // is cluster large enough?
            if((pts.size() / static_cast<FLOAT_TYPE>(n_pixels)) < minFraction)
                continue;

            const cv::Mat cluster = cv::Mat(pts).reshape(1);

            cv::Mat covariance;
            const cv::Mat mean = centers.row(static_cast<int>(id));

            //cv::calcCovarMatrix(cluster, covariance, mean, CV_COVAR_NORMAL|CV_COVAR_ROWS|CV_COVAR_USE_AVG, CV_32FC1);
            cv::calcCovarMatrix(cluster, covariance, mean, CV_COVAR_SCALE | CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_USE_AVG, CV_32FC1);

            MixtureComponent component;
            component.mean = mean;
            component.covariance = covariance + params.noise_factor * cv::Mat::eye(3, 3, CV_32FC1);
            component.iCovariance = component.covariance.inv();
            //component.weight = pts.size() / n_pixels;
            component.mass = pts.size();

            mixture.push_back(component);
            n_used_pixels += pts.size();
        }

        // compute weights
        for(int i = 0; i < mixture.size(); i++)
            mixture[i].weight = mixture[i].mass / static_cast<FLOAT_TYPE>(n_used_pixels);

        return mixture;
    }

    std::vector<MixtureComponent> gmm;
    size_t n_components;
    Parameters params;

public:
    bool initialized;
};

}