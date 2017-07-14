#include "VideoSegmenter.h"

using namespace ROAM;

// -----------------------------------------------------------------------------------
VideoSegmenter::VideoSegmenter(const VideoSegmenter::Params& params)
// -----------------------------------------------------------------------------------
{
    this->params = params;
    this->contour_init = false;
    this->frame_counter = 0;
    this->write_masks = false;
	this->current_contour_cost = std::numeric_limits<FLOAT_TYPE>::max();
}

// -----------------------------------------------------------------------------------
void VideoSegmenter::SetNextImageAuto(const cv::Mat &next_image)
// -----------------------------------------------------------------------------------
{
    if (!this->next_image.empty())
        this->prev_image = this->next_image.clone();
    else
        this->prev_image = next_image.clone();

    this->next_image = next_image.clone();
}

// -----------------------------------------------------------------------------------
void VideoSegmenter::SetNextImage(const cv::Mat &next_image)
// -----------------------------------------------------------------------------------
{
    this->next_image = next_image.clone();
}

// -----------------------------------------------------------------------------------
void VideoSegmenter::SetPrevImage(const cv::Mat &prev_image)
// -----------------------------------------------------------------------------------
{
    this->prev_image = prev_image.clone();
}

// -----------------------------------------------------------------------------------
void VideoSegmenter::SetParameters(const Params &params)
// -----------------------------------------------------------------------------------
{
    this->params = params;
}

// -----------------------------------------------------------------------------------
void VideoSegmenter::Write(cv::FileStorage& fs) const
// -----------------------------------------------------------------------------------
{
    this->params.Write(fs);
}

// -----------------------------------------------------------------------------------
static std::vector<FLOAT_TYPE> compDiffs(const std::shared_ptr<ClosedContour>& cont)
// -----------------------------------------------------------------------------------
{
    std::vector<FLOAT_TYPE> diffs;

    FLOAT_TYPE avg = 0.0f;
    for(auto itc = cont->contour_nodes.begin(); itc != --cont->contour_nodes.end(); ++itc)
    {
        auto nitc = std::next(itc, 1);
        const FLOAT_TYPE dis = static_cast<FLOAT_TYPE>(cv::norm(itc->GetCoordinates()-nitc->GetCoordinates()));
        avg += dis;
        diffs.push_back(dis);
    }
    avg /= static_cast<FLOAT_TYPE>(cont->contour_nodes.size());

    diffs.push_back( avg );

    return diffs;
}

// -----------------------------------------------------------------------------------
std::vector<cv::Point> VideoSegmenter::findDiffsContourMove(const std::vector<cv::Point> &move_to_point) const
// -----------------------------------------------------------------------------------
{
    //move_to point is the intermediate_contour
    std::vector<cv::Point> diff;
    diff.reserve(move_to_point.size());

    size_t ind = 0;
    for(auto it = contour->contour_nodes.begin(); it != contour->contour_nodes.end(); ++it, ++ind)
        diff.push_back(move_to_point[ind] - it->GetCoordinates());

    // outputs the vector that goes from contour coordinates to the new position
    return diff;
}

// -----------------------------------------------------------------------------------
void VideoSegmenter::performIntermediateContourMove(const std::vector<cv::Point> &move_to_point) const
// -----------------------------------------------------------------------------------
{
    std::vector<cv::Point> diff;
    diff.reserve(move_to_point.size());

    size_t ind = 0;
    for(auto it = contour->contour_nodes.begin(); it != contour->contour_nodes.end(); ++it)
        it->SetCoordinates(move_to_point[ind++]);
}

// -----------------------------------------------------------------------------------
static void
contourToMask(const std::vector<cv::Point> &contour_pts, cv::Mat &mask, cv::Size size)
// -----------------------------------------------------------------------------------
{
    // Mask from contours
    std::vector<std::vector<cv::Point>> array_conts = {contour_pts};
    mask = cv::Mat::zeros(size, CV_8UC1);
    cv::fillPoly(mask, array_conts, cv::Scalar(255));
}

// -----------------------------------------------------------------------------------
template<typename T> static
size_t addNodeToList(std::list<T>& added_list, const T &element, const int prev_node)
// -----------------------------------------------------------------------------------
{
    if(prev_node < 0)
    {
        added_list.push_back(element);
        return added_list.size();
    }
    else
    {
        added_list.insert(std::next(added_list.begin(), prev_node + 1), element);
        return prev_node + 1;
    }
}

// -----------------------------------------------------------------------------------
void VideoSegmenter::reinitializeNode(Node &node, const Node &next_node, int prev_node, int node_idx, ContourElementsHelper &ce) const
// -----------------------------------------------------------------------------------
{
    if(this->contour_init)
    {
        node = Node(node.GetCoordinates(), Node::Params(params.label_space_side));

        if (params.use_gradients_unary)
            node.AddUnaryTerm(ce.contour_gradient_unary);

        if (params.use_norm_pairwise)
            node.AddPairwiseTerm(ce.contour_norm_pairwise);

        if(params.use_temp_norm_pairwise)
        {
            const cv::Ptr<ROAM::TempAnglePairwise> tempAnglePairwise = ROAM::TempAnglePairwise::createPairwiseTerm(ROAM::TempAnglePairwise::Params(this->params.temp_angle_weight));
            const cv::Ptr<ROAM::TempNormPairwise> tempNormPairwise = ROAM::TempNormPairwise::createPairwiseTerm(ROAM::TempNormPairwise::Params(this->params.temp_norm_weight));

            tempAnglePairwise->Init(cv::Mat(cv::Mat_<FLOAT_TYPE>(1,1) << -5.f * CV_PI), cv::Mat());
            tempNormPairwise->Init(cv::Mat(cv::Mat_<FLOAT_TYPE>(1,1) << -1.f), cv::Mat());

            addNodeToList(ce.tempangle_pairwises_per_contour_node, tempAnglePairwise, prev_node);
            addNodeToList(ce.tempnorm_pairwises_per_contour_node, tempNormPairwise, prev_node);

            node.AddPairwiseTerm(tempAnglePairwise);
            node.AddPairwiseTerm(tempNormPairwise);

            const cv::Point a = node.GetCoordinates();
            const cv::Point b = next_node.GetCoordinates();

            addNodeToList(ce.prev_angles, FLOAT_TYPE(std::atan2(b.y - a.y, b.x - a.x)), prev_node);
            addNodeToList(ce.prev_norms, FLOAT_TYPE(cv::norm(b - a)), prev_node);
        }

        if(params.use_landmarks)
        {
            const cv::Ptr<ROAM::DistanceUnary> distanceUnary = ROAM::DistanceUnary::createUnaryTerm(
                                ROAM::DistanceUnary::Params(this->params.landmark_to_node_weight));
            distanceUnary->Init(cv::Mat());
            distanceUnary->SetPastDistance(-1.f /*when a negative number is passed, the cost becomes 0: only for intiialization*/);

            addNodeToList(ce.distance_unaries_per_contour_node, distanceUnary, prev_node);
            node.AddUnaryTerm(distanceUnary);
        }

        if(params.use_green_theorem_term)
        {
            const cv::Ptr<ROAM::GreenTheoremPairwise> gtPairwise =
                    ROAM::GreenTheoremPairwise::createPairwiseTerm(
                    ROAM::GreenTheoremPairwise::Params(this->params.green_theorem_weight,
                                                           this->contour->IsCounterClockWise(),
                                                           prev_node, node_idx));

            gtPairwise->Init(this->next_image, this->integral_negative_ratio_foreground_background_likelihood);

            addNodeToList(ce.green_theorem_pairwises, gtPairwise, prev_node);
            node.AddPairwiseTerm(gtPairwise);
        }

        if (params.use_snapcut_pairwise)
        {
            ROAM::SnapcutPairwise::Params snapcut_params(params.snapcut_weight,
                                                         params.snapcut_sigma_color, params.snapcut_region_height,
                                                         params.label_space_side, params.snapcut_number_clusters, true);

            const cv::Ptr<ROAM::SnapcutPairwise> snapcutPairwise = ROAM::SnapcutPairwise::createPairwiseTerm(snapcut_params);
            snapcutPairwise->Init(this->next_image, this->frame_mask/*intermediate_mask*/);

            addNodeToList(ce.snapcut_pairwises_per_contour_node, snapcutPairwise, prev_node);

            // I guesss I don't need to check this since frame_mask was already updated:
            // if (!params.use_landmarks || !this->landmarks_tree->DPTableIsBuilt())
            snapcutPairwise->InitializeEdge(node.GetCoordinates(),
                                            next_node.GetCoordinates(),
                                            this->next_image, this->frame_mask);

            node.AddPairwiseTerm(snapcutPairwise);
        }
    }

}

// -----------------------------------------------------------------------------------
template<typename T> static std::vector<T*>
createVectorOfPointersFromList(std::list<T>& the_list)
// -----------------------------------------------------------------------------------
{
    std::vector<T*> ptrs;
    for(auto it = the_list.begin(); it != the_list.end(); ++it)
	    ptrs.push_back(&(*it));

    return ptrs;
}

// -----------------------------------------------------------------------------------
void VideoSegmenter::SetContours(const std::vector<cv::Point> &contour_pts_)
// -----------------------------------------------------------------------------------
{
    std::vector<cv::Point> contour_pts;
    if(contour_pts_.empty() && !contour->contour_nodes.empty())
    {
        for(auto itc=contour->contour_nodes.begin(); itc!=this->contour->contour_nodes.end(); ++itc)
            contour_pts.push_back(itc->GetCoordinates());
    }
    else
    {
        contour_pts = contour_pts_;
    }

    chrono_timer_per_frame.Start();

    if (!this->contour_init)
    {
        this->params.Print();

        this->frame_mask = cv::Mat::zeros(this->next_image.size(), CV_8UC1);
        this->contour = std::make_shared<ClosedContour>(ClosedContour::Params(this->params.label_space_side * this->params.label_space_side));

        contourToMask(contour_pts, this->frame_mask, this->next_image.size());

        // Initializing position of nodes with contour_pts
        this->contour->contour_nodes.clear();
        for(size_t i=0; i<contour_pts.size(); ++i)
            this->contour->contour_nodes.push_back(ROAM::Node(contour_pts[i], Node::Params(params.label_space_side)));

        frame_counter=0;
        if(write_masks)
        {
            // Writing the first frame output
            cv::Mat draw = this->contour->DrawContour(this->prev_image, this->frame_mask);
            std::string filename = namefolder + std::string("/cont_") + std::to_string(frame_counter) + std::string(".png");
            cv::imwrite(filename, draw);

            filename = namefolder + std::string("/") + std::to_string(frame_counter) + std::string(".png");
            cv::imwrite(filename, frame_mask);
            ++frame_counter;
        }

        // Initialize Landmarks
        if(this->params.use_landmarks)
        {
            this->landmarks_tree = std::make_shared<StarGraph>(
                    StarGraph::Params(this->params.landmarks_searchspace_side,
                    this->params.landmark_min_response, this->params.landmark_max_area_overlap,
                    this->params.landmark_min_area, this->params.max_number_landmarks,
                    this->params.landmark_pairwise_weight));

            switch (this->params.warper_type)
            {
            case WARP_TRANSLATION:
                this->contour_warper = std::make_shared<RigidTransform_ContourWarper>(RigidTransform_ContourWarper::Params(RigidTransform_ContourWarper::TRANSLATION));
                break;

            default:
            case WARP_SIMILARITY:
                this->contour_warper = std::make_shared<RigidTransform_ContourWarper>(RigidTransform_ContourWarper::Params(RigidTransform_ContourWarper::SIMILARITY));
                break;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////// Processing Landmark Nodes ////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Create / Check for new  landmarks
    if(this->params.use_landmarks)
    {
        this->landmarks_tree->UpdateLandmarks(this->prev_image, this->frame_mask, true);
        this->landmarks_tree->TrackLandmarks(this->next_image); //landmarks only get moved when DP solved

        if(this->landmarks_tree->graph_nodes.size()>0)
        {
            this->landmarks_tree->BuildDPTable();

            // Run DP and apply moves
            FLOAT_TYPE min_cost_landmarks = 0.f;
            if(this->landmarks_tree->GetDPTable()->pairwise_costs.size() > 0)
            {
                min_cost_landmarks = this->landmarks_tree->RunDPInference();
                this->landmarks_tree->ApplyMoves();
            }
            LOG_INFO("VideoSegmenter::SetContours() - Landmarks min_cost: " << min_cost_landmarks);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////// Processing Intermediate Mask ////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    std::vector<Node*> contour_nodes_ptrs = createVectorOfPointersFromList(contour->contour_nodes);
    cv::Mat intermediate_mask;
    std::vector<cv::Point> intermediate_contour;
    std::vector<cv::Point> intermediate_motion_diff;
    if (this->params.use_landmarks && this->landmarks_tree->DPTableIsBuilt())
    {
        const std::shared_ptr<RigidTransform_ContourWarper> &warper = std::dynamic_pointer_cast<RigidTransform_ContourWarper>(this->contour_warper);
        warper->Init(this->landmarks_tree->correspondences_a, this->landmarks_tree->correspondences_b,this->frame_mask);
        intermediate_contour = warper->Warp(this->contour->contour_nodes);

        intermediate_motion_diff = findDiffsContourMove(intermediate_contour);
        contourToMask(intermediate_contour, intermediate_mask, this->next_image.size());
    }
    else
        intermediate_mask = this->frame_mask;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////// Processing Contour Nodes /////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Check if Green theorem is needed
    if (params.use_green_theorem_term)
    {
        if (!contour_init)
            fb_model = GlobalModel(GlobalModel::Params(3,50.f,50.f,1000.f));

        this->updateIntegralNegRatioForGreenTheorem(intermediate_mask);
    }

    // Simple terms if/else
    if (!contour_init)
    {
        if (params.use_gradients_unary)
        {
            contour_elements.contour_gradient_unary = ROAM::GradientUnary::createUnaryTerm(
                            ROAM::GradientUnary::Params(params.grad_type, params.grad_kernel_size, params.gradients_weight,
                                                        params.gaussian_smoothing_factor));
            contour_elements.contour_gradient_unary->Init(this->next_image);
        }

        if (params.use_norm_pairwise)
        {
            contour_elements.contour_norm_pairwise = ROAM::NormPairwise::createPairwiseTerm(ROAM::NormPairwise::Params(params.norm_type, params.norm_weight));
            contour_elements.contour_norm_pairwise->Init(cv::Mat(), cv::Mat());
        }

        if (params.use_gradient_pairwise && params.use_gradients_unary)
        {
            contour_elements.contour_generic_pairwise = ROAM::GenericPairwise::createPairwiseTerm(ROAM::GenericPairwise::Params(params.gradient_pairwise_weight));
            contour_elements.contour_generic_pairwise->Init(contour_elements.contour_gradient_unary->GetUnaries(), cv::Mat());
        }

    }
    else
    {

        if (params.use_gradients_unary)
            contour_elements.contour_gradient_unary->Update(this->next_image);

        if (params.use_gradient_pairwise && params.use_gradients_unary)
            contour_elements.contour_generic_pairwise->Update(contour_elements.contour_gradient_unary->GetUnaries(), cv::Mat());

        if (params.use_norm_pairwise)
            contour_elements.contour_norm_pairwise->Update(cv::Mat(), cv::Mat());
    }

    // Resize lists of pairwises
    if (!contour_init)
    {
        if (params.use_snapcut_pairwise)
        {
            contour_elements.snapcut_pairwises_per_contour_node = std::list<cv::Ptr<ROAM::SnapcutPairwise>>(contour_nodes_ptrs.size());
        }

        if (params.use_temp_norm_pairwise)
        {
            contour_elements.tempangle_pairwises_per_contour_node = std::list<cv::Ptr<ROAM::TempAnglePairwise>>(contour_nodes_ptrs.size());
            contour_elements.tempnorm_pairwises_per_contour_node = std::list<cv::Ptr<ROAM::TempNormPairwise>>(contour_nodes_ptrs.size());
            contour_elements.prev_norms = std::list<FLOAT_TYPE>(contour_nodes_ptrs.size(), -1.f);
            contour_elements.prev_angles = std::list<FLOAT_TYPE>(contour_nodes_ptrs.size(), static_cast<FLOAT_TYPE>(-5 * CV_PI));
        }

        if (params.use_landmarks)
            contour_elements.distance_unaries_per_contour_node = std::list<cv::Ptr<ROAM::DistanceUnary>>(contour_nodes_ptrs.size());

        if (params.use_green_theorem_term)
            contour_elements.green_theorem_pairwises = std::list<cv::Ptr<ROAM::GreenTheoremPairwise>>(contour_nodes_ptrs.size());

    }

    // For simplicity: Let us first process non-cuda terms
    if (!contour_init)
    {
        #pragma omp parallel for
        for(auto ii = 0; ii<contour_nodes_ptrs.size(); ++ii)
        {
            if (params.use_gradients_unary)
                contour_nodes_ptrs[ii]->AddUnaryTerm(contour_elements.contour_gradient_unary);

            if (params.use_norm_pairwise)
                contour_nodes_ptrs[ii]->AddPairwiseTerm(contour_elements.contour_norm_pairwise);

            if (params.use_temp_norm_pairwise)
            {
                auto tempnorm_pairwises_per_contour_node_it = std::next(contour_elements.tempnorm_pairwises_per_contour_node.begin(), ii);
                auto tempangle_pairwises_per_contour_node_it = std::next(contour_elements.tempangle_pairwises_per_contour_node.begin(), ii);
                auto prev_norms_it = std::next(contour_elements.prev_norms.begin(), ii);
                auto prev_angles_it = std::next(contour_elements.prev_angles.begin(), ii);

                const cv::Ptr<ROAM::TempAnglePairwise> tempAnglePairwise = ROAM::TempAnglePairwise::createPairwiseTerm(ROAM::TempAnglePairwise::Params(this->params.temp_angle_weight));
                const cv::Ptr<ROAM::TempNormPairwise> tempNormPairwise = ROAM::TempNormPairwise::createPairwiseTerm(ROAM::TempNormPairwise::Params(this->params.temp_norm_weight));

                tempAnglePairwise->Init( cv::Mat(cv::Mat_<FLOAT_TYPE>(1,1)<<*prev_angles_it), cv::Mat() );
                tempNormPairwise->Init( cv::Mat(cv::Mat_<FLOAT_TYPE>(1,1)<<*prev_norms_it), cv::Mat() );

                *tempangle_pairwises_per_contour_node_it = tempAnglePairwise;
                *tempnorm_pairwises_per_contour_node_it = tempNormPairwise;

                contour_nodes_ptrs[ii]->AddPairwiseTerm(tempAnglePairwise);
                contour_nodes_ptrs[ii]->AddPairwiseTerm(tempNormPairwise);

                if (ii == contour_nodes_ptrs.size()-1)
                {
                    const cv::Point a = contour_nodes_ptrs[ii]->GetCoordinates();
                    const cv::Point b = contour_nodes_ptrs[0]->GetCoordinates();
                    *prev_angles_it = static_cast<FLOAT_TYPE>(std::atan2(b.y - a.y, b.x - a.x));
                    *prev_norms_it = static_cast<FLOAT_TYPE>(cv::norm(b - a));
                }
                else
                {
                    const cv::Point a = contour_nodes_ptrs[ii]->GetCoordinates();
                    const cv::Point b = contour_nodes_ptrs[ii+1]->GetCoordinates();
                    *prev_angles_it = static_cast<FLOAT_TYPE>(std::atan2(b.y - a.y, b.x - a.x));
                    *prev_norms_it = static_cast<FLOAT_TYPE>(cv::norm(b - a));
                }
            }

            if (params.use_landmarks)
            {
                auto distance_unaries_per_contour_node_it = std::next(contour_elements.distance_unaries_per_contour_node.begin(), ii);
                cv::Ptr<ROAM::DistanceUnary> distanceUnary = ROAM::DistanceUnary::createUnaryTerm(
                            ROAM::DistanceUnary::Params(this->params.landmark_to_node_weight) );
                distanceUnary->Init( this->landmarks_tree->VectorOfClosestLandmarkPoints(contour_nodes_ptrs[ii]->GetCoordinates(),
                                     this->params.landmark_to_node_radius) );

                distanceUnary->SetPastDistance(-1.f /*when a negative number is passed, the cost becomes 0: only for intiialization*/);
                *distance_unaries_per_contour_node_it = distanceUnary;
                contour_nodes_ptrs[ii]->AddUnaryTerm(distanceUnary);
            }

            if (params.use_green_theorem_term)
            {
                auto green_pairwises_per_contour_node_it = std::next(contour_elements.green_theorem_pairwises.begin(), ii);
                const cv::Ptr<ROAM::GreenTheoremPairwise> gtPairwise =
                        ROAM::GreenTheoremPairwise::createPairwiseTerm(
                            ROAM::GreenTheoremPairwise::Params(this->params.green_theorem_weight,
                                                               this->contour->IsCounterClockWise(),
                                                               ii, (ii+1==contour_nodes_ptrs.size())?0:ii+1));
                //std::cout<<integral_negative_ratio_foreground_background_likelihood<<std::endl;
                gtPairwise->Init(this->next_image, this->integral_negative_ratio_foreground_background_likelihood);

                *green_pairwises_per_contour_node_it = gtPairwise;
                contour_nodes_ptrs[ii]->AddPairwiseTerm(gtPairwise);
            }

#ifndef WITH_CUDA
            // Fill vector of snapcut terms (non cuda)
            if (params.use_snapcut_pairwise)
            {
                auto snapcut_pairwises_per_contour_node_it = std::next(contour_elements.snapcut_pairwises_per_contour_node.begin(), ii);
                ROAM::SnapcutPairwise::Params snapcut_params(params.snapcut_weight,
                                                             params.snapcut_sigma_color, params.snapcut_region_height,
                                                             params.label_space_side,
                                                             params.snapcut_number_clusters, false);

                cv::Ptr<ROAM::SnapcutPairwise> snapcutPairwise = ROAM::SnapcutPairwise::createPairwiseTerm(snapcut_params);
                snapcutPairwise->Init(this->next_image, intermediate_mask);

                if (!params.use_landmarks || !landmarks_tree->DPTableIsBuilt())
                {
                    if (ii == contour_nodes_ptrs.size()-1)
                        snapcutPairwise->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(), contour_nodes_ptrs[0]->GetCoordinates(),
                                prev_image, frame_mask);
                    else
                        snapcutPairwise->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(), contour_nodes_ptrs[ii+1]->GetCoordinates(),
                                 prev_image, frame_mask);
                    *(snapcut_pairwises_per_contour_node_it) = snapcutPairwise;
                    contour_nodes_ptrs[ii]->AddPairwiseTerm(*snapcut_pairwises_per_contour_node_it);
                }
                else
                {
                    if (ii == contour_nodes_ptrs.size()-1)
                        snapcutPairwise->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(), contour_nodes_ptrs[0]->GetCoordinates(),
                                prev_image, frame_mask,
                                intermediate_motion_diff[ii], intermediate_motion_diff[0]);
                    else
                        snapcutPairwise->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(), contour_nodes_ptrs[ii+1]->GetCoordinates(),
                                prev_image, frame_mask,
                                intermediate_motion_diff[ii], intermediate_motion_diff[ii+1]);
                    *snapcut_pairwises_per_contour_node_it = snapcutPairwise;
                    contour_nodes_ptrs[ii]->AddPairwiseTerm(*snapcut_pairwises_per_contour_node_it);
                }
            }
#endif
        }
    }
    else // The contour was already initialized, so we update
    {
        if (this->params.use_landmarks)
        {
            #pragma omp parallel for
            for(auto ii = 0; ii<contour_nodes_ptrs.size(); ++ii)
            {
                auto distance_unaries_per_contour_node_it = std::next(contour_elements.distance_unaries_per_contour_node.begin(), ii);

                const cv::Ptr<ROAM::DistanceUnary> &distanceUnary = *distance_unaries_per_contour_node_it;
                distanceUnary->Update( this->landmarks_tree->VectorOfClosestLandmarkPoints(contour_nodes_ptrs[ii]->GetCoordinates(),
                                                                                           this->params.landmark_to_node_radius) );
                if (!landmarks_tree->DPTableIsBuilt())
                    distanceUnary->SetPastDistance( -1.f );
                else
                    distanceUnary->SetPastDistance( this->landmarks_tree->AverageDistanceToClosestLandmarkPoints(contour_nodes_ptrs[ii]->GetCoordinates(),
                                                                                                                 this->params.landmark_to_node_radius) );
                    //distanceUnary->SetPastDistance( -1.f );
            }
        }

        if (params.use_temp_norm_pairwise)
        {
            #pragma omp parallel for
            for(auto ii = 0; ii<contour_nodes_ptrs.size(); ++ii)
            {
                auto tempnorm_pairwises_per_contour_node_it = std::next(contour_elements.tempnorm_pairwises_per_contour_node.begin(), ii);
                auto tempangle_pairwises_per_contour_node_it = std::next(contour_elements.tempangle_pairwises_per_contour_node.begin(), ii);
                auto prev_norms_it = std::next(contour_elements.prev_norms.begin(), ii);
                auto prev_angles_it = std::next(contour_elements.prev_angles.begin(), ii);

                const cv::Ptr<ROAM::TempAnglePairwise> &tempAnglePairwise = *tempangle_pairwises_per_contour_node_it;
                const cv::Ptr<ROAM::TempNormPairwise> &tempNormPairwise = *tempnorm_pairwises_per_contour_node_it;

                tempAnglePairwise->Update( cv::Mat(cv::Mat_<FLOAT_TYPE>(1,1)<<*prev_angles_it), cv::Mat() );
                tempNormPairwise->Update( cv::Mat(cv::Mat_<FLOAT_TYPE>(1,1)<<*prev_norms_it), cv::Mat() );

                if (ii == contour_nodes_ptrs.size()-1)
                {
                    const cv::Point a = contour_nodes_ptrs[ii]->GetCoordinates();
                    const cv::Point b = contour_nodes_ptrs[0]->GetCoordinates();
                    *prev_angles_it = static_cast<FLOAT_TYPE>(std::atan2(b.y - a.y, b.x - a.x));
                    *prev_norms_it = static_cast<FLOAT_TYPE>(cv::norm(b - a));
                }
                else
                {
                    const cv::Point a = contour_nodes_ptrs[ii]->GetCoordinates();
                    const cv::Point b = contour_nodes_ptrs[ii + 1]->GetCoordinates();
                    *prev_angles_it = static_cast<FLOAT_TYPE>(std::atan2(b.y - a.y, b.x - a.x));
                    *prev_norms_it = static_cast<FLOAT_TYPE>(cv::norm(b - a));
                }
            }
        }

        if (params.use_green_theorem_term)
        {
            #pragma omp parallel for
            for(auto ii = 0; ii<contour_nodes_ptrs.size(); ++ii)
            {
                auto green_pairwises_per_contour_node_it = std::next(contour_elements.green_theorem_pairwises.begin(), ii);
                const cv::Ptr<ROAM::GreenTheoremPairwise> &gtPairwise = *green_pairwises_per_contour_node_it;

                gtPairwise->Update(this->next_image, this->integral_negative_ratio_foreground_background_likelihood);
            }
        }

#ifndef WITH_CUDA
        #pragma omp parallel for
        for(auto ii = 0; ii<contour_nodes_ptrs.size(); ++ii)
        {
            // Fill vector of snapcut terms (non cuda)
            if (params.use_snapcut_pairwise)
            {
                auto snapcut_pairwises_per_contour_node_it = std::next(contour_elements.snapcut_pairwises_per_contour_node.begin(), ii);
                cv::Ptr<ROAM::SnapcutPairwise>& snapcutPairwise = *(snapcut_pairwises_per_contour_node_it);
                snapcutPairwise->Update(this->next_image, intermediate_mask);

                if (!params.use_landmarks || !landmarks_tree->DPTableIsBuilt())
                {
                    if (ii == contour_nodes_ptrs.size()-1)
                        snapcutPairwise->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(), contour_nodes_ptrs[0]->GetCoordinates(),
                                prev_image, frame_mask);
                    else
                        snapcutPairwise->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(), contour_nodes_ptrs[ii+1]->GetCoordinates(),
                                prev_image, frame_mask);
                }
                else
                {
                    if (ii == contour_nodes_ptrs.size()-1)
                        snapcutPairwise->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(), contour_nodes_ptrs[0]->GetCoordinates(),
                                prev_image, frame_mask,
                                intermediate_motion_diff[ii], intermediate_motion_diff[0]);
                    else
                        snapcutPairwise->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(), contour_nodes_ptrs[ii+1]->GetCoordinates(),
                                prev_image, frame_mask,
                                intermediate_motion_diff[ii], intermediate_motion_diff[ii+1]);
                }
            }
        }
#endif
    }

#ifdef WITH_CUDA
    contour->BuildDPTable();
    if (params.use_snapcut_pairwise)
    {
        if (!contour_init)
        {
            ROAM::SnapcutPairwise::Params snapcut_params(params.snapcut_weight,
                                                         params.snapcut_sigma_color, params.snapcut_region_height,
                                                         params.label_space_side, params.snapcut_number_clusters, true);

            #pragma omp parallel for
            for(auto ii = 0; ii<contour_nodes_ptrs.size(); ++ii)
            {
                auto snapcut_pairwises_per_contour_node_it = std::next(contour_elements.snapcut_pairwises_per_contour_node.begin(), ii);

                *snapcut_pairwises_per_contour_node_it = ROAM::SnapcutPairwise::createPairwiseTerm(snapcut_params);
                (*snapcut_pairwises_per_contour_node_it)->Init(this->next_image, intermediate_mask);

                if (!params.use_landmarks || !landmarks_tree->DPTableIsBuilt())
                {

                    if (ii == contour_nodes_ptrs.size()-1)
                        (*snapcut_pairwises_per_contour_node_it)->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(),
                                                                               contour_nodes_ptrs[0]->GetCoordinates(),
                                                                               prev_image, frame_mask);
                    else
                        (*snapcut_pairwises_per_contour_node_it)->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(),
                                                                               contour_nodes_ptrs[ii+1]->GetCoordinates(),
                                                                               prev_image, frame_mask);
                }
                else
                {
                    if (ii == contour_nodes_ptrs.size()-1)
                        (*snapcut_pairwises_per_contour_node_it)->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(),
                                                                               contour_nodes_ptrs[0]->GetCoordinates(),
                                                                               prev_image, frame_mask,
                                                                               intermediate_motion_diff[ii],
                                                                               intermediate_motion_diff[0]);
                    else
                        (*snapcut_pairwises_per_contour_node_it)->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(),
                                                                               contour_nodes_ptrs[ii+1]->GetCoordinates(),
                                                                               prev_image, frame_mask,
                                                                               intermediate_motion_diff[ii],
                                                                               intermediate_motion_diff[ii+1]);
                }

                contour_nodes_ptrs[ii]->AddPairwiseTerm(*snapcut_pairwises_per_contour_node_it);
            }
        }
        else
        {
            #pragma omp parallel for
            for(auto ii = 0; ii<contour_nodes_ptrs.size(); ++ii)
            {
                auto snapcut_pairwises_per_contour_node_it = std::next(contour_elements.snapcut_pairwises_per_contour_node.begin(), ii);

                cv::Ptr<ROAM::SnapcutPairwise>& snapcutPairwise = *snapcut_pairwises_per_contour_node_it;

                snapcutPairwise->Update(this->next_image, intermediate_mask);

                if (!params.use_landmarks || !landmarks_tree->DPTableIsBuilt())
                {
                    if (ii == contour_nodes_ptrs.size()-1)
                        snapcutPairwise->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(),
                                                        contour_nodes_ptrs[0]->GetCoordinates(), prev_image, frame_mask);
                    else
                        snapcutPairwise->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(),
                                                        contour_nodes_ptrs[ii+1]->GetCoordinates(), prev_image, frame_mask);
                }
                else
                {
                    if (ii == contour_nodes_ptrs.size()-1)
                        snapcutPairwise->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(),
                                                        contour_nodes_ptrs[0]->GetCoordinates(),
                                                        prev_image, frame_mask,
                                                        intermediate_motion_diff[ii],
                                                        intermediate_motion_diff[0]);
                    else
                        snapcutPairwise->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(),
                                                        contour_nodes_ptrs[ii+1]->GetCoordinates(),
                                                        prev_image, frame_mask,
                                                        intermediate_motion_diff[ii],
                                                        intermediate_motion_diff[ii+1]);
                }

                *snapcut_pairwises_per_contour_node_it = snapcutPairwise;
            }
        }
    }

    if (params.use_landmarks && landmarks_tree->DPTableIsBuilt())
        performIntermediateContourMove(intermediate_contour);

    if (params.use_snapcut_pairwise)
    {
        contour->ExecuteCudaPairwises(params.snapcut_region_height, params.snapcut_sigma_color,
                                      params.snapcut_weight, next_image.rows, next_image.cols);
    }

    contour->BuildDPTable();
#else
    if (params.use_landmarks && landmarks_tree->DPTableIsBuilt())
        performIntermediateContourMove(intermediate_contour);

    contour->BuildDPTable();
#endif

    this->contour_init = true;
}

// -----------------------------------------------------------------------------------
bool VideoSegmenter::IsInit() const
// -----------------------------------------------------------------------------------
{
    return this->contour_init;
}

// -----------------------------------------------------------------------------------
void VideoSegmenter::WriteOutput(const std::string &foldername)
// -----------------------------------------------------------------------------------
{
    this->namefolder = foldername;
    this->write_masks = true;
}

// -----------------------------------------------------------------------------------
std::vector<cv::Point> VideoSegmenter::ProcessFrame()
// -----------------------------------------------------------------------------------
{
    assert(this->contour_init);

    this->current_contour_cost = this->contour->RunDPInference();

    this->contour->ApplyMoves();

    std::vector<cv::Point> output_cont_pts;
    for (auto itc=contour->contour_nodes.begin(); itc!=this->contour->contour_nodes.end(); ++itc)
        output_cont_pts.push_back(itc->GetCoordinates());

    contourToMask(output_cont_pts, this->frame_mask, this->next_image.size());

    if (this->params.use_graphcut_term)
        automaticReparametrization();


    const double t = chrono_timer_per_frame.Stop();
    LOG_INFO("VideoSegmenter::ProcessFrame() - TOTAL per-frame exec: " << t);
    LOG_INFO("VideoSegmenter::ProcessFrame() - min cost: " << this->current_contour_cost );

    costs_per_frame.push_back(this->current_contour_cost);

    return output_cont_pts;
}

// -----------------------------------------------------------------------------------
cv::Mat VideoSegmenter::WritingOperations()
// -----------------------------------------------------------------------------------
{
    if (write_masks)
    {
        const cv::Mat draw = this->contour->DrawContour(this->next_image, this->frame_mask);
        std::string filename = namefolder + std::string("/cont_") + std::to_string(frame_counter) + std::string(".png");
        cv::imwrite(filename, draw);

        filename = namefolder + std::string("/") + std::to_string(frame_counter) + std::string(".png");
        cv::imwrite(filename, frame_mask);

        filename = namefolder + std::string("/cont_") + std::to_string(frame_counter) + std::string(".txt");
        this->WriteTxt(filename);

        ++frame_counter;
    }
    return frame_mask;
}

// -----------------------------------------------------------------------------------
void VideoSegmenter::WriteTxt(const std::string &filename) const
// -----------------------------------------------------------------------------------
{
	std::ofstream f_out(filename);

	const auto &nodes = this->contour->contour_nodes;
	f_out << nodes.size() << " # number of nodes (for closed contour)" <<std::endl;
	f_out << "x y" << std::endl;

	for (auto it = nodes.begin(); it != nodes.end(); ++it)
		f_out << it->GetCoordinates().x << " " << it->GetCoordinates().y << std::endl;


	if(this->params.use_landmarks)
	{
		f_out << "------------------------------------------------" << std::endl;

		const auto &landmarks = this->landmarks_tree->graph_nodes;
		
		f_out << std::endl << landmarks.size() << " # number of landmarks (pictorial structure)" << std::endl;
		f_out << "x y" << std::endl;

		for (auto it = landmarks.begin(); it != landmarks.end(); ++it)
		{
			const cv::Point n_c = GET_NODE_FROM_TUPLE(*it)->GetCoordinates();
			f_out << n_c.x << " " << n_c.y << std::endl;
		}

	}

}

// -----------------------------------------------------------------------------------
VideoSegmenter::Params VideoSegmenter::getParams() const
// -----------------------------------------------------------------------------------
{
    return params;
}

// -----------------------------------------------------------------------------------
void VideoSegmenter::setParams(const VideoSegmenter::Params &value)
// -----------------------------------------------------------------------------------
{
    params = value;
}

// -----------------------------------------------------------------------------------
static
std::vector<cv::Point> ContourToVectorPoints(const std::shared_ptr<ClosedContour> &contour)
// -----------------------------------------------------------------------------------
{
    std::vector<cv::Point> pts;

    for (auto it = contour->contour_nodes.begin(); it != contour->contour_nodes.end(); ++it)
        pts.push_back(it->GetCoordinates());

    return pts;
}

// -----------------------------------------------------------------------------------
static
cv::Rect FindContourRange(const std::vector<cv::Point> &contour, int slack=30)
// -----------------------------------------------------------------------------------
{
    cv::Rect range;

    int min_x=100000;
    int min_y=100000;
    int max_x=0;
    int max_y=0;
    for (int p_i = 0; p_i < contour.size(); ++p_i)
    {
        const cv::Point& p = contour[p_i];


        min_x = std::min(p.x,min_x);
        max_x = std::max(p.x,max_x);
        min_y = std::min(p.y,min_y);
        max_y = std::max(p.y,max_y);
    }

    range = cv::Rect(min_x-slack/2, min_y-slack/2, max_x-min_x+slack, max_y-min_y+slack);
    return range;
}


// -----------------------------------------------------------------------------------
void VideoSegmenter::automaticReparametrization()
// -----------------------------------------------------------------------------------
{
    std::vector<cv::Point> next_contour = ContourToVectorPoints(this->contour);

    LOG_INFO("VideoSegmenter::AutomaticReparametrization() - Reparametrizing ");

    cv::Mat gc_segmented;

    bool keep_checking_for_proposals = true;

    int iter_while = 0;
    const int MAX_REPARAMETRIZATION_ITERS = 1;
    while (keep_checking_for_proposals && iter_while<MAX_REPARAMETRIZATION_ITERS)
    {
        keep_checking_for_proposals = false; // if I accept a proposal, I'll set it somewhere inside the loop

        if (!this->graphcut.Initialized())
            this->graphcut = GC_Energy(GC_Energy::Params(1,50.f,50.f,1000.f));

        std::vector<ROAM::ProposalsBox> proposals;
        cv::Mat contour_lmap = contour->GenerateLikelihoodMap(this->next_image);

        const int slack = 50;
        cv::Rect range = FindContourRange(next_contour, slack) &
                cv::Rect(0, 0, this->next_image.cols, this->next_image.rows);
        if (!this->params.use_green_theorem_term) //!> use internal model (initialized and updated inside function
        {
            gc_segmented = this->graphcut.Segment(this->next_image, this->frame_mask,
                                                  next_contour, contour_lmap, GlobalModel(),
                                                  range);

        }
        else //!> use green theorem model (avoids double computation)
        {

            gc_segmented = this->graphcut.Segment(this->next_image, this->frame_mask,
                                                  next_contour, contour_lmap, this->fb_model, range);
        }

        // At this point we have the two segmentation masks:
        // From GCUT: gc_segmented, from ROAM: frame_mask
        // Find contour of largest blob in gc_segmented and the blob itself
        std::vector<cv::Point> gc_contour = ContourFromMask(gc_segmented);
        cv::Mat gc_largest_blob;
        contourToMask(gc_contour, gc_largest_blob, gc_segmented.size());

        findDifferences(gc_segmented, gc_contour, gc_largest_blob,
                        next_contour, this->frame_mask,
                        proposals);

        if(proposals.size() == 0)
        {
            LOG_INFO("VideoSegmenter::AutomaticReparametrization() - No proposals. ");
            return;
        }

        // sort and start from the largest one
        std::sort(proposals.begin(), proposals.end(), [](ProposalsBox const& a, ProposalsBox const& b) { return a.mass > b.mass; });

        // propose move and compute new energy
        for(size_t i = 0; i < proposals.size(); ++i)
        {
            const std::set<size_t> &blob_remove = proposals[i].remove_nodes;
            const std::vector<cv::Point> &blob_add = proposals[i].add_nodes;

            std::vector<bool> toRemove(next_contour.size(), false);
            size_t n_nodes_to_remove = 0;
            for(std::set<size_t>::const_iterator it = blob_remove.begin(); it != blob_remove.end(); ++it)
            {
                toRemove[*it] = true;
                n_nodes_to_remove++;
            }

            if(n_nodes_to_remove == toRemove.size())
                continue; // not goona start removing if the proposal says i need to remove everuthing

            std::shared_ptr<ClosedContour> proposed_contour = std::make_shared<ClosedContour>(*(this->contour));
            ContourElementsHelper proposed_ce = this->contour_elements;
            std::list<Node> &proposed_nodes = proposed_contour->contour_nodes;

            std::vector<Node*> proposed_nodes_ptrs = createVectorOfPointersFromList(proposed_nodes);

            int min_id = static_cast<int>(proposals[i].min_max_ids.first);
            const int max_id = static_cast<int>(proposals[i].min_max_ids.second);

            bool positive_orientation = true; 
            if(blob_add.size() > 0)
            {
                const FLOAT_TYPE pos_distance = static_cast<FLOAT_TYPE>(cv::norm(blob_add[0] - proposed_nodes_ptrs[min_id]->GetCoordinates()) + cv::norm(blob_add[blob_add.size() - 1] - proposed_nodes_ptrs[max_id]->GetCoordinates()));
                const FLOAT_TYPE neg_distance = static_cast<FLOAT_TYPE>(cv::norm(blob_add[0] - proposed_nodes_ptrs[max_id]->GetCoordinates()) + cv::norm(blob_add[blob_add.size() - 1] - proposed_nodes_ptrs[min_id]->GetCoordinates()));
                positive_orientation = pos_distance < neg_distance ? true : false;
            }

            if(toRemove[min_id])
                min_id = std::max(min_id - 1, 0);

            proposed_contour->PruneNodes(toRemove);
            proposed_ce.PruneContourElements(toRemove, params.use_landmarks, params.use_snapcut_pairwise,
                                             params.use_temp_norm_pairwise, params.use_green_theorem_term);

            proposed_nodes_ptrs =  createVectorOfPointersFromList(proposed_nodes);

            const FLOAT_TYPE step_size = 5.f;
            // add new nodes
            // A small number of nodes (and >0): Bigger proposals slow down too much the execution
            if(blob_add.size() > 0 && blob_add.size()<next_contour.size()/3 && blob_remove.size()<next_contour.size()/4 && blob_remove.size()>0)
            {
                size_t iter = 0;
                if(positive_orientation)
                    for(std::vector<cv::Point>::const_iterator it = blob_add.begin(); it != blob_add.end(); ++it)
                    {
                        const cv::Point &prev = proposed_nodes_ptrs[min_id + iter]->GetCoordinates();
                        const cv::Point &curr = *it;
                        if(std::sqrt(static_cast<FLOAT_TYPE>((prev.x - curr.x)*(prev.x - curr.x) + (prev.y - curr.y)*(prev.y - curr.y))) > step_size)
                        {
                            size_t added_node_idx = proposed_contour->AddNode(*it, min_id + static_cast<int>(iter)); // check this (adding in the right place?)
                            proposed_nodes_ptrs =  createVectorOfPointersFromList(proposed_nodes);
                            auto it_node = std::next(proposed_contour->contour_nodes.begin(), added_node_idx);
                            auto it_next = std::next(it_node, 1);

                            if (it_next!=--proposed_contour->contour_nodes.end())
                                reinitializeNode(*it_node, *it_next, static_cast<int>(added_node_idx - 1), static_cast<int>(added_node_idx), proposed_ce);
                            else
                                reinitializeNode(*it_node, *proposed_contour->contour_nodes.begin(), static_cast<int>(added_node_idx - 1), 0, proposed_ce);

                            ++iter;
                        }
                    }
                else 
                    for(std::vector<cv::Point>::const_reverse_iterator it = blob_add.rbegin(); it != blob_add.rend(); ++it)
                    {
                        const cv::Point &prev = proposed_nodes_ptrs[min_id + iter]->GetCoordinates();
                        const cv::Point &curr = *it;
                        if(std::sqrt(static_cast<FLOAT_TYPE>((prev.x - curr.x)*(prev.x - curr.x) + (prev.y - curr.y)*(prev.y - curr.y))) >  step_size)
                        {
                            size_t added_node_idx = proposed_contour->AddNode(*it, min_id + static_cast<int>(iter));
                            proposed_nodes_ptrs =  createVectorOfPointersFromList(proposed_nodes);

                            auto it_node = std::next(proposed_contour->contour_nodes.begin(), added_node_idx);
                            auto it_next = std::next(it_node, 1);

                            if (it_next!=--proposed_contour->contour_nodes.end())
                                reinitializeNode(*it_node, *it_next, static_cast<int>(added_node_idx - 1), static_cast<int>(added_node_idx), proposed_ce);
                            else
                                reinitializeNode(*it_node, *proposed_contour->contour_nodes.begin(), static_cast<int>(added_node_idx - 1), 0, proposed_ce);

                            ++iter;
                        }
                    }

                if(proposed_nodes.size() < 50)
                    continue;


                // evaluate energy...
                const FLOAT_TYPE proposed_energy = proposed_contour->GetTotalContourCost(
                                        params.snapcut_region_height, params.snapcut_sigma_color,
                                        params.snapcut_weight, next_image.rows, next_image.cols,
                                        this->params.use_snapcut_pairwise );

                if(proposed_energy < (1.f-this->params.reparametrization_failsafe)*this->current_contour_cost)
                {
                    this->contour = proposed_contour;
                    this->current_contour_cost = proposed_energy;
                    this->contour_elements = proposed_ce;
                    next_contour = ContourToVectorPoints(proposed_contour);
                    contourToMask(next_contour, this->frame_mask, this->frame_mask.size());

                    keep_checking_for_proposals = true;

                    break;
                }
            } // end if blob add

        }// endfor: propose move and compute new energy
        ++iter_while;

    }// end while
}


// -----------------------------------------------------------------------------------
void VideoSegmenter::updateIntegralNegRatioForGreenTheorem(const cv::Mat &mask)
// -----------------------------------------------------------------------------------
{
    if (!fb_model.initialized)
        this->fb_model.Initialize(this->prev_image, mask);
    else
        this->fb_model.Update(this->prev_image, mask);

    this->fb_model.ComputeLikelihood(this->next_image, this->global_foreground_likelihood, this->global_background_likelihood);

    // compute negative log fg/bg ratio
    const cv::Mat ratio =
        this->global_foreground_likelihood / (this->global_background_likelihood + std::numeric_limits<FLOAT_TYPE>::epsilon());

    cv::Mat ratio_cost;
    cv::log(ratio, ratio_cost);

    cv::Mat neg_ratio = -ratio_cost;

    this->integral_negative_ratio_foreground_background_likelihood =
            this->accumulate(neg_ratio);
}

// -----------------------------------------------------------------------------------
cv::Mat VideoSegmenter::accumulate(const cv::Mat &input) const
// -----------------------------------------------------------------------------------
{
    assert(input.type() == CV_32FC1);

    cv::Mat output = cv::Mat::zeros(input.size(), input.type());

    // copy first row
    input.row(0).copyTo(output.row(0));

    // per-column line integrals
    #pragma omp parallel for
    for(auto x = 0; x < input.cols; ++x)
        for(auto y = 1; y < input.rows; ++y)
            output.at<float>(y, x) = input.at<float>(y, x) + output.at<float>(y - 1, x);

    return output;
}

// -----------------------------------------------------------------------------------
void VideoSegmenter::ContourElementsHelper::PruneContourElements(const std::vector<bool> &remove, bool use_du, bool use_sc,
        bool use_tn, bool use_gt)
// -----------------------------------------------------------------------------------
{
    auto it_du = this->distance_unaries_per_contour_node.begin();
    auto it_pa = this->prev_angles.begin();
    auto it_pn = this->prev_norms.begin();
    auto it_sc = this->snapcut_pairwises_per_contour_node.begin();
    auto it_tn = this->tempnorm_pairwises_per_contour_node.begin();
    auto it_ta = this->tempangle_pairwises_per_contour_node.begin();
    auto it_gt = this->green_theorem_pairwises.begin();

    std::list<bool> remove_list;
    std::copy( remove.begin(), remove.end(), std::back_inserter( remove_list ) );

    auto it_rm = remove_list.begin();

    while (it_rm != remove_list.end())
    {
        bool is_rem = *it_rm;
        if (is_rem)
        {
            it_rm = remove_list.erase(it_rm);
            if (use_du)
                it_du = distance_unaries_per_contour_node.erase(it_du);

            if (use_tn)
                it_pa = prev_angles.erase(it_pa);

            if (use_tn)
                it_pn = prev_norms.erase(it_pn);

            if (use_sc)
                it_sc = snapcut_pairwises_per_contour_node.erase(it_sc);

            if (use_tn)
                it_tn = tempnorm_pairwises_per_contour_node.erase(it_tn);

            if (use_tn)
                it_ta = tempangle_pairwises_per_contour_node.erase(it_ta);

            if (use_gt)
                it_gt = green_theorem_pairwises.erase(it_gt);
        }
        else
        {
            ++it_rm;

            if (use_du)
                ++it_du;

            if (use_tn)
                ++it_pa;

            if (use_tn)
                ++it_pn;

            if (use_sc)
                ++it_sc;

            if (use_tn)
                ++it_tn;

            if (use_tn)
                ++it_ta;

            if (use_gt)
                ++it_gt;
        }
    }
}

// -----------------------------------------------------------------------------------
void VideoSegmenter::Params::Read(cv::FileStorage &fs)
// -----------------------------------------------------------------------------------
{
    int buffer;

    if(!fs["label_space_side"].empty())
    {
        fs["label_space_side"] >> buffer;
        this->label_space_side = buffer;
    }

    if(!fs["use_gradients_unary"].empty())
        fs["use_gradients_unary"] >> this->use_gradients_unary;

    if(!fs["grad_type"].empty())
    {
        fs["grad_type"] >> buffer;
        this->grad_type = static_cast<GradientUnary::GradType>(buffer);
    }

    if (!fs["use_norm_pairwise"].empty())
        fs["use_norm_pairwise"] >> this->use_norm_pairwise;

    if (!fs["norm_type"].empty())
    {
        fs["norm_type"] >> buffer;
        this->norm_type = static_cast<NormPairwise::NormType>(buffer);
    }

    if (!fs["norm_weight"].empty())
        fs["norm_weight"] >> this->norm_weight;

    if (!fs["use_temp_norm_pairwise"].empty())
        fs["use_temp_norm_pairwise"] >> this->use_temp_norm_pairwise;

    if (!fs["temp_norm_weight"].empty())
        fs["temp_norm_weight"] >> this->temp_norm_weight;

    if (!fs["temp_angle_weight"].empty())
        fs["temp_angle_weight"] >> this->temp_angle_weight;

    if(!fs["use_snapcut_pairwise"].empty())
        fs["use_snapcut_pairwise"] >> this->use_snapcut_pairwise;

    if(!fs["snapcut_region_height"].empty())
        fs["snapcut_region_height"] >> this->snapcut_region_height;

    if(!fs["snapcut_sigma_color"].empty())
        fs["snapcut_sigma_color"] >> this->snapcut_sigma_color;

    if(!fs["snapcut_number_clusters"].empty())
        fs["snapcut_number_clusters"] >> this->snapcut_number_clusters;

    if(!fs["snapcut_weight"].empty())
        fs["snapcut_weight"] >> this->snapcut_weight;

    if (!fs["use_landmarks"].empty())
       fs["use_landmarks"] >> this->use_landmarks;

    if (!fs["max_number_landmarks"].empty())
        fs["max_number_landmarks"] >> this->max_number_landmarks;

    if (!fs["landmark_max_area_overlap"].empty())
        fs["landmark_max_area_overlap"] >> this->landmark_max_area_overlap;

    if (!fs["landmark_min_area"].empty())
        fs["landmark_min_area"] >> this->landmark_min_area;

    if (!fs["landmark_min_response"].empty())
        fs["landmark_min_response"] >> this->landmark_min_response;

    if (!fs["landmark_pairwise_weight"].empty())
        fs["landmark_pairwise_weight"] >> this->landmark_pairwise_weight;

    if (!fs["landmark_to_node_weight"].empty())
        fs["landmark_to_node_weight"] >> this->landmark_to_node_weight;

    if (!fs["landmark_to_node_radius"].empty())
        fs["landmark_to_node_radius"] >> this->landmark_to_node_radius;

    if (!fs["landmarks_searchspace_side"].empty())
        fs["landmarks_searchspace_side"] >> this->landmarks_searchspace_side;

    if (!fs["use_gradient_pairwise"].empty())
        fs["use_gradient_pairwise"] >> this->use_gradient_pairwise;

    if (!fs["gradient_pairwise_weight"].empty())
        fs["gradient_pairwise_weight"] >> this->gradient_pairwise_weight;

    if (!fs["warper_type"].empty())
    {
        fs["warper_type"] >> buffer;
        this->warper_type = static_cast<WarpType>(buffer);
    }

    if (!fs["use_green_theorem_term"].empty())
       fs["use_green_theorem_term"] >> this->use_green_theorem_term;

    if (!fs["green_theorem_weight"].empty())
       fs["green_theorem_weight"] >> this->green_theorem_weight;

    if (!fs["use_graphcut_term"].empty())
       fs["use_graphcut_term"] >> this->use_graphcut_term;

    if (!fs["reparametrization_failsafe"].empty())
        fs["reparametrization_failsafe"] >> this->reparametrization_failsafe;
}

// -----------------------------------------------------------------------------------
void VideoSegmenter::Params::Write(cv::FileStorage &fs) const
// -----------------------------------------------------------------------------------
{
    fs << "label_space_side" << static_cast<int>(this->label_space_side);

    fs << "use_gradients_unary" << this->use_gradients_unary;
    fs << "gaussian_smoothing_factor" << this->gaussian_smoothing_factor;
    fs << "gradients_weight" << this->gradients_weight;
    fs << "grad_kernel_size" << this->grad_kernel_size;
    fs << "grad_type" << static_cast<int>(this->grad_type);

    fs << "use_norm_pairwise" << this->use_norm_pairwise;
    fs << "norm_weight" << this->norm_weight;
    fs << "norm_type" << static_cast<int>(this->norm_type);

    fs << "use_temp_norm_pairwise" << this->use_temp_norm_pairwise;
    fs << "temp_angle_weight" << this->temp_angle_weight;
    fs << "temp_norm_weight" << this->temp_norm_weight;

    fs << "use_snapcut_pairwise" << this->use_snapcut_pairwise;
    fs << "snapcut_region_height" << this->snapcut_region_height;
    fs << "snapcut_sigma_color" << this->snapcut_sigma_color;
    fs << "snapcut_weight" << this->snapcut_weight;
    fs << "snapcut_number_clusters" << this->snapcut_number_clusters;

    fs << "use_landmarks" << this->use_landmarks;
    fs << "max_number_landmarks" << this->max_number_landmarks;
    fs << "landmark_max_area_overlap" << this->landmark_max_area_overlap;
    fs << "landmark_min_area" << this->landmark_min_area;
    fs << "landmark_min_response" << this->landmark_min_response;
    fs << "landmark_pairwise_weight" << this->landmark_pairwise_weight;
    fs << "landmark_to_node_weight" << this->landmark_to_node_weight;
    fs << "landmark_to_node_radius" << this->landmark_to_node_radius;
    fs << "landmarks_searchspace_side" << this->landmarks_searchspace_side;

    fs << "use_gradient_pairwise" << this->use_gradient_pairwise;
    fs << "gradient_pairwise_weight" << this->gradient_pairwise_weight;

    fs << "warper_type" << this->warper_type;

    fs << "use_green_theorem_term" << this->use_green_theorem_term;
    fs << "green_theorem_weight" << this->green_theorem_weight;
    fs << "use_graphcut_term" << this->use_graphcut_term;
    fs << "reparametrization_failsafe" << this->reparametrization_failsafe;
}

// -----------------------------------------------------------------------------------
void VideoSegmenter::Params::Print() const
// -----------------------------------------------------------------------------------
{
    LOG_INFO("");
    LOG_INFO( "======================= USED PARAMETERS ========================" );
    LOG_INFO( "label_space_side: " << static_cast<int>(this->label_space_side) );

    LOG_INFO( "use_gradients_unary: " << this->use_gradients_unary );
    LOG_INFO( "gaussian_smoothing_factor: " << this->gaussian_smoothing_factor );
    LOG_INFO( "gradients_weight: " << this->gradients_weight );
    LOG_INFO( "grad_kernel_size: " << this->grad_kernel_size );
    LOG_INFO( "grad_type: " << static_cast<int>(this->grad_type) );

    LOG_INFO( "use_norm_pairwise: " << this->use_norm_pairwise );
    LOG_INFO( "norm_weight: " << this->norm_weight );
    LOG_INFO( "norm_type: " << static_cast<int>(this->norm_type) );

    LOG_INFO( "use_temp_norm_pairwise: " << this->use_temp_norm_pairwise );
    LOG_INFO( "temp_angle_weight: " << this->temp_angle_weight );
    LOG_INFO( "temp_norm_weight: " << this->temp_norm_weight );

    LOG_INFO( "use_snapcut_pairwise: " << this->use_snapcut_pairwise );
    LOG_INFO( "snapcut_region_height: " << this->snapcut_region_height );
    LOG_INFO( "snapcut_sigma_color: " << this->snapcut_sigma_color );
    LOG_INFO( "snapcut_weight: " << this->snapcut_weight );
    LOG_INFO( "snapcut_number_clusters: " << this->snapcut_number_clusters );

    LOG_INFO( "use_landmarks: " << this->use_landmarks );
    LOG_INFO( "max_number_landmarks: " << this->max_number_landmarks );
    LOG_INFO( "landmark_max_area_overlap: " << this->landmark_max_area_overlap );
    LOG_INFO( "landmark_min_area: " << this->landmark_min_area );
    LOG_INFO( "landmark_min_response: " << this->landmark_min_response );
    LOG_INFO( "landmark_pairwise_weight: " << this->landmark_pairwise_weight );
    LOG_INFO( "landmark_to_node_weight: " << this->landmark_to_node_weight );
    LOG_INFO( "landmark_to_node_radius: " << this->landmark_to_node_radius );
    LOG_INFO( "landmarks_searchspace_side: " << this->landmarks_searchspace_side );

    LOG_INFO( "use_gradient_pairwise: " << this->use_gradient_pairwise );
    LOG_INFO( "gradient_pairwise_weight: " << this->gradient_pairwise_weight );

    LOG_INFO( "warper_type: " << this->warper_type );

    LOG_INFO( "use_green_theorem_term: " << this->use_green_theorem_term );
    LOG_INFO( "green_theorem_weight: " << this->green_theorem_weight );
    LOG_INFO( "use_graphcut_term: " << this->use_graphcut_term );
    LOG_INFO( "reparametrization_failsafe: " << this->reparametrization_failsafe );

    LOG_INFO( "================================================================" );
    LOG_INFO("");
}