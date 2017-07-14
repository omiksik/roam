#include "Reparametrization.h"

namespace ROAM
{

// TODO: Reparametrization was quickly implemented during a short layover at the Munich Airport. 
//    The code is very sub-optimal and should be re-implemented.

// -----------------------------------------------------------------------------------
std::vector<cv::Point> ContourFromMask(const cv::Mat &mask, const int type_simplification)
// -----------------------------------------------------------------------------------
{
    std::vector<cv::Point> output_contour;

    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(mask.clone(), contours, CV_RETR_EXTERNAL, type_simplification);

    // Choose largest contour
    std::vector<double> sizes;
    sizes.resize(contours.size());

    #pragma omp parallel for
    for(auto i = 0; i < contours.size(); ++i)
        sizes[i] = cv::contourArea(contours[i]);

    const auto maxIndex = std::distance(sizes.begin(), std::max_element(sizes.begin(), sizes.end()));

    if (type_simplification == CV_CHAIN_APPROX_NONE && contours.size()>0 )
        for ( size_t i=0; i<contours[maxIndex].size(); i=i+2 ) // TODO: not sure why ``i+2'', isn't this hardcoded quantization?
            output_contour.push_back(contours[maxIndex][i]);
    else
        output_contour = contours[maxIndex];

    return output_contour;
}

// -----------------------------------------------------------------------------------
std::vector< std::vector<cv::Point> > ComponentsFromMask(const cv::Mat &mask, const int type_simplification)
// -----------------------------------------------------------------------------------
{
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours( mask, contours, CV_RETR_EXTERNAL, type_simplification );
    return contours;
}

// -----------------------------------------------------------------------------------
static
size_t findClosestNode(const std::vector<cv::Point> &node_pts, const cv::Point &pt,
                       const std::vector<cv::Point> &banned,
                       const FLOAT_TYPE banned_dst = 1.0)
// -----------------------------------------------------------------------------------
{
    std::vector<FLOAT_TYPE> distances = std::vector<FLOAT_TYPE>(node_pts.size(), std::numeric_limits<FLOAT_TYPE>::infinity());

    #pragma omp parallel for
    for(auto i = 0; i < node_pts.size(); ++i)
    {
        for(auto j = 0; j < banned.size(); ++j)
            if(cv::norm(banned[j] - node_pts[i]) < banned_dst)
                continue;

        distances[i] = static_cast<FLOAT_TYPE>(cv::norm(node_pts[i] - pt));
    }

    const size_t min_id = std::distance(distances.begin(), std::min_element(distances.begin(), distances.end()));

    return min_id;
}

// -----------------------------------------------------------------------------------
static
size_t findClosestNode(const std::vector<cv::Point> &node_pts, const cv::Point &pt)
// -----------------------------------------------------------------------------------
{
    std::vector<FLOAT_TYPE> distances;
    distances.resize(node_pts.size());

    #pragma omp parallel for
    for(auto i = 0; i < node_pts.size(); ++i)
        distances[i] = static_cast<FLOAT_TYPE>(cv::norm(node_pts[i] - pt));

    const size_t min_id = std::distance(distances.begin(), std::min_element(distances.begin(), distances.end()));

    return min_id;
}

// -----------------------------------------------------------------------------------
static
void findPoint(const std::vector<cv::Point2i> &pts, const cv::Point2i &pt,
               std::set<size_t> &ids)
// -----------------------------------------------------------------------------------
{
    for(size_t i = 0; i < pts.size(); ++i)
        if(pts[i] == pt)
            ids.insert(i);
}

// -----------------------------------------------------------------------------------
static
FLOAT_TYPE computePixelDistance(const std::vector<cv::Point> &proposed_contour,
                                const cv::Point &a, const cv::Point &b, const bool pos,
                                std::vector<size_t> &path)
// -----------------------------------------------------------------------------------
{
    std::set<size_t> id_a;
    std::set<size_t> id_b;

    findPoint(proposed_contour, a, id_a);
    findPoint(proposed_contour, b, id_b);

    if(id_a.empty() || id_b.empty())
        return -1;

    FLOAT_TYPE distance = std::numeric_limits<FLOAT_TYPE>::infinity();

    for(std::set<size_t>::const_iterator it1 = id_a.begin(); it1 != id_a.end(); ++it1)
        for(std::set<size_t>::const_iterator it2 = id_b.begin(); it2 != id_b.end(); ++it2)
        {
            // TODO: change
            const size_t idx_a = *it1;
            const size_t idx_b = *it2;

            const int min_id = static_cast<int>(std::min(idx_a, idx_b));
            const int max_id = static_cast<int>(std::max(idx_a, idx_b));

            const FLOAT_TYPE current_dst = pos ? static_cast<float>(std::abs(max_id - min_id)) : static_cast<float>(proposed_contour.size() - std::abs(max_id - min_id));

            if(current_dst < distance)
            {
                distance = current_dst;

                // TODO: reimplement
                std::set<size_t> added;

                if(pos)
                {
                    path.clear();
                    path.reserve(max_id - min_id);

                    for(size_t i = min_id; i < max_id; ++i)
                        if(!(std::find(added.begin(), added.end(), i) != added.end()))
                        {
                            path.push_back(i);
                            added.insert(i);    // is this just to remove potential duplicates?
                        }
                }
                else
                {
                    path.clear();
                    path.reserve(proposed_contour.size() - max_id + min_id);

                    for(size_t i = max_id; i < proposed_contour.size(); ++i)
                        if(!(std::find(added.begin(), added.end(), i) != added.end()))
                        {
                            path.push_back(i);
                            added.insert(i);
                        }

                    for(size_t i = 0; i < min_id; ++i)
                        if(!(std::find(added.begin(), added.end(), i) != added.end()))
                        {
                            path.push_back(i);
                            added.insert(i);
                        }
                }
            }
        }

    return distance;
}

// -----------------------------------------------------------------------------------
bool getDifferences(const cv::Mat &gcut_dst_tf, const cv::Mat &contour_dst_tf,
                    const std::vector<cv::Point> &contour, const std::vector<cv::Point> &blob,
                    std::set<size_t> &to_remove, std::vector<cv::Point> &to_add, size_t &min_id, size_t &max_id)
// -----------------------------------------------------------------------------------
{
    ROAM::Timer timer;
    timer.Start();

    const FLOAT_TYPE max_dst_tf = 1.0;
    const FLOAT_TYPE max_Node_dst_tf = 2.0;

    // add nodes - let's swap it with finding of endpoints
    cv::Mat tst = cv::Mat::zeros(gcut_dst_tf.size(), CV_8UC1);
    #pragma omp parallel for
    for(auto j = 0; j < blob.size(); ++j)
        tst.at<uchar>(blob[j]) = 255; // TODO: change to pointer?

    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(tst, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    std::sort(contours.begin(), contours.end(), [](std::vector<cv::Point> const& a, std::vector<cv::Point> const& b) { return a.size() > b.size(); });

    if(contours.size() == 0)
        return false;

    std::vector<cv::Point> new_i_points;

    // find points close to endpoints
    for(size_t j = 0; j < contours[0].size(); ++j)
        if(contour_dst_tf.at<float>(contours[0][j]) < max_dst_tf && gcut_dst_tf.at<float>(contours[0][j]) < max_dst_tf)
            new_i_points.push_back(contours[0][j]);

    if(new_i_points.size() == 0)
        return false;

    cv::Point new_a = new_i_points[0];
    cv::Point new_b = new_i_points[0];

    std::vector<size_t> proposed_to_add;
    std::vector<size_t> proposed_to_remove;

    FLOAT_TYPE new_max_dst = 0;

    // find the correct endpoints
    for(size_t j = 0; j < new_i_points.size(); ++j)
        for(size_t k = j; k < new_i_points.size(); ++k)
        {
            const cv::Point &tmp_a = new_i_points[j];
            const cv::Point &tmp_b = new_i_points[k];

            // extract segments
            std::vector<size_t> pos_path;
            const FLOAT_TYPE pos_curr_dst = computePixelDistance(contours[0], tmp_a, tmp_b, true, pos_path);

            std::vector<size_t> neg_path;
            const FLOAT_TYPE neg_curr_dst = computePixelDistance(contours[0], tmp_a, tmp_b, false, neg_path);

            // determine how far they are from the proposed contour
            FLOAT_TYPE pos_tf_dst = 0.0;
            for(std::vector<size_t>::const_iterator it = pos_path.begin(); it != pos_path.end(); ++it)
                pos_tf_dst += gcut_dst_tf.at<float>(contours[0][*it]);

            FLOAT_TYPE neg_tf_dst = 0.0;
            for(std::vector<size_t>::const_iterator it = neg_path.begin(); it != neg_path.end(); ++it)
                neg_tf_dst += gcut_dst_tf.at<float>(contours[0][*it]);

            // determine which one is proposed
            std::vector<size_t> tmp_proposed_to_add;
            std::vector<size_t> tmp_proposed_to_remove;
            FLOAT_TYPE curr_dst;
            if(pos_tf_dst < neg_tf_dst)
            {
                tmp_proposed_to_add = pos_path;
                tmp_proposed_to_remove = neg_path;
                curr_dst = pos_curr_dst;
            }
            else
            {
                tmp_proposed_to_add = neg_path;
                tmp_proposed_to_remove = pos_path;
                curr_dst = neg_curr_dst;
            }

            const FLOAT_TYPE new_curr_dst = static_cast<float>(cv::norm(tmp_a - tmp_b));
            if(new_curr_dst >= new_max_dst)
            {
                new_a = tmp_a;
                new_b = tmp_b;

                proposed_to_add = tmp_proposed_to_add;
                proposed_to_remove = tmp_proposed_to_remove;

                new_max_dst = new_curr_dst;
            }
        }

    for(std::vector<size_t>::const_iterator it = proposed_to_add.begin(); it != proposed_to_add.end(); ++it)
        if(contour_dst_tf.at<float>(contours[0][*it]) > 5.0)
            to_add.push_back(contours[0][*it]);

    const size_t id_a = findClosestNode(contour, new_a, to_add);
    const size_t id_b = findClosestNode(contour, new_b, to_add);

    to_remove.insert(id_a);
    to_remove.insert(id_b);

    for(std::vector<size_t>::const_iterator it = proposed_to_remove.begin(); it != proposed_to_remove.end(); ++it)
    {
        const size_t id = findClosestNode(contour, contours[0][*it]);
        const FLOAT_TYPE dst = static_cast<float>(cv::norm(contours[0][*it] - contour[id]));
        if(dst < max_Node_dst_tf)
            to_remove.insert(id);
    }

    if(to_remove.empty())
        return false;

    // add (potentially) missing/not detected points
    const int set_min_id = static_cast<int>(*(to_remove.begin()));
    const int set_max_id = static_cast<int>(*(to_remove.rbegin()));

    // TODO: be careful with id close to 0
    for(int id = set_min_id + 1; id < set_max_id - 1; ++id)
        to_remove.insert(id);

    min_id = std::max(int(*std::min_element(to_remove.begin(), to_remove.end())) - 1, 0);
    max_id = *std::max_element(to_remove.begin(), to_remove.end());

    return true;
}

// -----------------------------------------------------------------------------------
std::vector<cv::Mat> findDifferences(const cv::Mat &gc_segmented, 
                                     const std::vector<cv::Point> &gc_contour, 
                                     const cv::Mat &gc_largest_blob,
                                     const std::vector<cv::Point> &curr_contour, 
                                     const cv::Mat &curr_mask,
                                     std::vector<ProposalsBox> &all_proposals)
// -----------------------------------------------------------------------------------
{

    ROAM::Timer timer;
    timer.Start();

    std::vector<cv::Mat> outputs;

    // Find residuals (Holes and Missing parts)
    const cv::Mat gcut_contour = gc_largest_blob - curr_mask;
    const cv::Mat contour_gcut = curr_mask - gc_largest_blob;

    // ProposalBlobs
    ROAM::ProposalsBlobs proposals;

    // repar_holes filter out
    {
        // distance TF of contour
        cv::Mat boundary = cv::Mat::zeros(gc_segmented.rows, gc_segmented.cols, CV_8UC1);
        cv::polylines(boundary, curr_contour, true, cv::Scalar(255, 255, 255));
        cv::distanceTransform(255 - boundary, proposals.dst_c_g, CV_DIST_L2, 3);

        std::vector<std::vector<cv::Point> > tmp_blobs = ComponentsFromMask(contour_gcut);

        for(size_t i = 0; i < tmp_blobs.size(); ++i)
        {
            const std::vector<cv::Point> &blob = tmp_blobs[i];

            FLOAT_TYPE min_val = std::numeric_limits<FLOAT_TYPE>::infinity();
            FLOAT_TYPE max_val = 0.0;
            for(size_t j = 0; j < blob.size(); ++j)
            {
                const cv::Point2i &pt = blob[j];
                min_val = std::min(min_val, proposals.dst_c_g.at<float>(pt));
                max_val = std::max(max_val, proposals.dst_c_g.at<float>(pt));
                if(!(min_val > 1.0) && max_val > 2.0)
                    break;
            }

            if(max_val > 2.0 && min_val > 1.0)
                proposals.blobs_holes.push_back(blob);
            else if(max_val > 2.0)
                proposals.blobs_c_g.push_back(blob);

        }
    }

    // missing_parts filter out
    {
        cv::Mat boundary = cv::Mat::zeros(gc_segmented.rows, gc_segmented.cols, CV_8UC1);
        cv::polylines(boundary, gc_contour, true, cv::Scalar(255, 255, 255));
        cv::distanceTransform(255 - boundary, proposals.dst_g_c, CV_DIST_L2, 3);

        std::vector<std::vector<cv::Point> > tmp_blobs = ComponentsFromMask(gcut_contour);

        for(size_t i = 0; i < tmp_blobs.size(); ++i)
        {
            const std::vector<cv::Point2i> &blob = tmp_blobs[i];
            bool valid = false;
            for(size_t j = 0; j < blob.size(); ++j)
            {
                const cv::Point2i &pt = blob[j];
                if(proposals.dst_g_c.at<float>(pt) > 2.0)
                {
                    valid = true;
                    break;
                }
            }

            if(valid)
                proposals.blobs_g_c.push_back(blob);
        }
    }

    // g - c
    #if 0
    for(size_t i = 0; i < proposals.blobs_g_c.size(); ++i)
    {
        const std::vector<cv::Point> &blob = proposals.blobs_g_c[i];

        std::set<size_t> to_remove;
        std::vector<cv::Point> to_add;
        size_t min_id, max_id;
        const bool valid = getDifferences(proposals.dst_g_c, proposals.dst_c_g, curr_contour, blob, to_remove, to_add, min_id, max_id);
        if(valid)
        {

            ROAM::ProposalsBox current_proposal;
            current_proposal.remove_nodes = to_remove;
            current_proposal.add_nodes = to_add;
            current_proposal.min_max_ids = std::pair<size_t, size_t>(min_id, max_id);
            current_proposal.mass = to_remove.size();

            all_proposals.push_back(current_proposal);
            if(min_id > 10000)
            {
                throw;
            }
        }
    }
    #endif

    {
        std::vector<bool> valid(proposals.blobs_g_c.size(), false);
        std::vector<ProposalsBox> tmp_proposals(proposals.blobs_g_c.size());

        #pragma omp parallel for
        for(auto i = 0; i < proposals.blobs_g_c.size(); ++i)
        {
            const std::vector<cv::Point> &blob = proposals.blobs_g_c[i];

            ProposalsBox &tmp_proposal = tmp_proposals[i];
            valid[i] = ROAM::getDifferences(proposals.dst_g_c, proposals.dst_c_g, curr_contour, blob, 
                                      tmp_proposal.remove_nodes, tmp_proposal.add_nodes, tmp_proposal.min_max_ids.first, tmp_proposal.min_max_ids.second);
            
            tmp_proposal.mass = tmp_proposal.remove_nodes.size();
        }

        for(size_t i = 0; i < proposals.blobs_g_c.size(); ++i)
            if(valid[i])
            {
                all_proposals.push_back(tmp_proposals[i]);
                if(tmp_proposals[i].min_max_ids.first > 10000)
                    throw;
            }
    }

    // c - g
    #if 0
    for(size_t i = 0; i < proposals.blobs_c_g.size(); ++i)
    {
        const std::vector<cv::Point> &blob = proposals.blobs_c_g[i];

        std::set<size_t> to_remove;
        std::vector<cv::Point> to_add;
        size_t min_id, max_id;
        const bool valid = getDifferences(proposals.dst_g_c, proposals.dst_c_g, curr_contour, blob, to_remove, to_add, min_id, max_id);
        if(valid)
        {
            ProposalsBox current_proposal;
            current_proposal.remove_nodes = to_remove;
            current_proposal.add_nodes = to_add;
            current_proposal.min_max_ids = std::pair<size_t, size_t>(min_id, max_id);
            current_proposal.mass = to_remove.size();

            all_proposals.push_back(current_proposal);
            if(min_id > 10000)
            {
                throw;
            }
        }
    }
    #endif

    {
        std::vector<bool> valid(proposals.blobs_c_g.size(), false);
        std::vector<ProposalsBox> tmp_proposals(proposals.blobs_c_g.size());

        #pragma omp parallel for
        for(auto i = 0; i < proposals.blobs_c_g.size(); ++i)
        {
            const std::vector<cv::Point> &blob = proposals.blobs_c_g[i];

            ProposalsBox &tmp_proposal = tmp_proposals[i];
            valid[i] = ROAM::getDifferences(proposals.dst_g_c, proposals.dst_c_g, curr_contour, blob, 
                                      tmp_proposal.remove_nodes, tmp_proposal.add_nodes, tmp_proposal.min_max_ids.first, tmp_proposal.min_max_ids.second);

            tmp_proposal.mass = tmp_proposal.remove_nodes.size();
        }

        for(size_t i = 0; i < proposals.blobs_c_g.size(); ++i)
            if(valid[i])
            {
                all_proposals.push_back(tmp_proposals[i]);
                if(tmp_proposals[i].min_max_ids.first > 10000)
                    throw;
            }

    }

// TODO: I think we haven't used the holes, so I'm commenting this out
#if 0
    // holes
    for(size_t i = 0; i < proposals.blobs_holes.size(); ++i)
    {
        const std::vector<cv::Point> &blob = proposals.blobs_holes[i];

        cv::Mat blob_contour = cv::Mat::zeros(gc_segmented.rows, gc_segmented.cols, CV_8UC1);

        std::vector<std::vector<cv::Point> > all_changes;
        cv::connectedComponents(blob_contour, all_changes);

        for(size_t idx = 0; idx < all_changes.size(); ++idx)
        {
            const std::vector<cv::Point> &line = all_changes[idx];

            if(line.size() < 2)
                continue;

            std::vector<cv::Point> approx;
            cv::approxPolyDP(line, approx, 0.05, false);

            if(approx.size() < 2)
                continue;

            proposals.contour_holes.push_back(approx);
        }

    }
#endif

return outputs;
}

// -----------------------------------------------------------------------------------
size_t addedNodesKeepTrack(const int prev_node, std::list<bool>& added)
// -----------------------------------------------------------------------------------
{
    if(prev_node < 0)
    {
        added.push_back(true);
        return added.size();
    }
    else
    {
        added.insert(std::next(added.begin(), prev_node + 1), true);
        return prev_node + 1;
    }
}

}