#include "DynamicProgramming.h"
#include "ClosedContour.h"
#include "EnergyTerms.h"
#include "Node.h"
#include "RotatedRect.h"
#include "ContourWarper.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

TEST(DPTest, OpenChainSimple)
{
    const int num_labels=2, num_nodes=3;
    std::shared_ptr<roam::OpenChainDPTable> dp_table = std::make_shared<roam::OpenChainDPTable>(num_labels, num_nodes);
    roam::DPTableUnaries& unary_costs = dp_table->unary_costs;
    roam::DPTablePairwises& pairwise_costs = dp_table->pairwise_costs;

    unary_costs = { {2.0f, 1.0f}, {3.0f, 0.1f}, {0.0f, 0.0f} };

    pairwise_costs = { { {2.0f,1.0f},{0.5f,3.0f} },
                       { {0.5f,0.5f},{0.5f,0.2f} } };

    roam::ChainDP dp;
    FLOAT_TYPE min_cost;
    std::vector<roam::label> results = dp.Minimize(dp_table, min_cost);
    std::vector<roam::label> ground_truth = {0,1,1};

    ASSERT_TRUE(results.size()==ground_truth.size());

    for (size_t i=0; i<results.size(); ++i)
    {
       ASSERT_EQ(ground_truth[i], results[i]);
    }

    const FLOAT_TYPE gt_cost = 3.3f;
    ASSERT_FLOAT_EQ(min_cost, gt_cost);
}

TEST(DPTest, OpenChain)
{
    const int num_labels=5, num_nodes=6;
    std::shared_ptr<roam::OpenChainDPTable> dp_table = std::make_shared<roam::OpenChainDPTable>(num_labels, num_nodes);
    roam::DPTableUnaries& unary_costs = dp_table->unary_costs;
    roam::DPTablePairwises& pairwise_costs = dp_table->pairwise_costs;

    // Example from the book "Computer vision: models, learning and inference" (Version of July 7, 2012) by "S. J.D. Prince", pages 248-250.
    unary_costs = { {2.0f, 0.8f, 4.3f, 6.4f, 2.3f}, {1.1f, 4.8f, 2.3f, 0.0f, 2.2f}, {5.7f, 1.0f, 2.4f, 6.1f, 4.9f},
                    {1.5f, 3.0f, 2.4f, 0.8f, 8.9f}, {6.0f, 6.9f, 6.6f, 7.1f, 1.0f}, {3.1f, 3.3f, 6.2f, 2.1f, 9.8f} };

    const FLOAT_TYPE INF = std::numeric_limits<FLOAT_TYPE>::infinity();
    pairwise_costs = { { {0.0f,2.0f,INF,INF,INF},{2.0f,0.0f,2.0f,INF,INF},{INF,2.0f,0.0f,2.0f,INF},{INF,INF,2.0f,0.0f,2.0f},{INF,INF,INF,2.0f,0.0f} },
                       { {0.0f,2.0f,INF,INF,INF},{2.0f,0.0f,2.0f,INF,INF},{INF,2.0f,0.0f,2.0f,INF},{INF,INF,2.0f,0.0f,2.0f},{INF,INF,INF,2.0f,0.0f} },
                       { {0.0f,2.0f,INF,INF,INF},{2.0f,0.0f,2.0f,INF,INF},{INF,2.0f,0.0f,2.0f,INF},{INF,INF,2.0f,0.0f,2.0f},{INF,INF,INF,2.0f,0.0f} },
                       { {0.0f,2.0f,INF,INF,INF},{2.0f,0.0f,2.0f,INF,INF},{INF,2.0f,0.0f,2.0f,INF},{INF,INF,2.0f,0.0f,2.0f},{INF,INF,INF,2.0f,0.0f} },
                       { {0.0f,2.0f,INF,INF,INF},{2.0f,0.0f,2.0f,INF,INF},{INF,2.0f,0.0f,2.0f,INF},{INF,INF,2.0f,0.0f,2.0f},{INF,INF,INF,2.0f,0.0f} } };

    roam::ChainDP dp;
    FLOAT_TYPE min_cost;
    std::vector<roam::label> results = dp.Minimize(dp_table, min_cost);
    std::vector<roam::label> ground_truth = {1, 2, 2, 3, 4, 3};

    //std::cerr<<"The cost was: "<<min_cost<<std::endl;

    ASSERT_TRUE(results.size()==ground_truth.size());

    //TODO: Replace by better MACRO
    for (size_t i=0; i<results.size(); ++i)
    {
       ASSERT_EQ(ground_truth[i], results[i]);
    }

    const FLOAT_TYPE gt_cost = 17.4f;
    ASSERT_FLOAT_EQ(min_cost, gt_cost);
}

TEST(DPTest, ClosedChain)
{
    const int num_labels=5, num_nodes=6;
    std::shared_ptr<roam::ClosedChainDPTable> dp_table = std::make_shared<roam::ClosedChainDPTable>(num_labels, num_nodes);
    roam::DPTableUnaries& unary_costs = dp_table->unary_costs;
    roam::DPTablePairwises& pairwise_costs = dp_table->pairwise_costs;

    unary_costs = { {2.0f, 0.8f, 4.3f, 6.4f, 2.3f}, {1.1f, 4.8f, 2.3f, 0.0f, 2.2f}, {5.7f, 1.0f, 2.4f, 6.1f, 4.9f},
                    {1.5f, 3.0f, 2.4f, 0.8f, 8.9f}, {6.0f, 6.9f, 6.6f, 7.1f, 1.0f}, {3.1f, 3.3f, 6.2f, 2.1f, 9.8f} };

    const FLOAT_TYPE INF = std::numeric_limits<FLOAT_TYPE>::infinity();
    pairwise_costs = { { {0.0f,2.0f,INF,INF,INF},{2.0f,0.0f,2.0f,INF,INF},{INF,2.0f,0.0f,2.0f,INF},{INF,INF,2.0f,0.0f,2.0f},{INF,INF,INF,2.0f,0.0f} },
                       { {0.0f,2.0f,INF,INF,INF},{2.0f,0.0f,2.0f,INF,INF},{INF,2.0f,0.0f,2.0f,INF},{INF,INF,2.0f,0.0f,2.0f},{INF,INF,INF,2.0f,0.0f} },
                       { {0.0f,2.0f,INF,INF,INF},{2.0f,0.0f,2.0f,INF,INF},{INF,2.0f,0.0f,2.0f,INF},{INF,INF,2.0f,0.0f,2.0f},{INF,INF,INF,2.0f,0.0f} },
                       { {0.0f,2.0f,INF,INF,INF},{2.0f,0.0f,2.0f,INF,INF},{INF,2.0f,0.0f,2.0f,INF},{INF,INF,2.0f,0.0f,2.0f},{INF,INF,INF,2.0f,0.0f} },
                       { {0.0f,2.0f,INF,INF,INF},{2.0f,0.0f,2.0f,INF,INF},{INF,2.0f,0.0f,2.0f,INF},{INF,INF,2.0f,0.0f,2.0f},{INF,INF,INF,2.0f,0.0f} },
                       { {0.0f,2.0f,INF,INF,INF},{2.0f,0.0f,2.0f,INF,INF},{INF,2.0f,0.0f,2.0f,INF},{INF,INF,2.0f,0.0f,2.0f},{INF,INF,INF,2.0f,0.0f} }
                     };

    roam::ClosedChainDP dp;
    FLOAT_TYPE min_cost;
    std::vector<roam::label> results = dp.Minimize(dp_table, min_cost);
    std::vector<roam::label> ground_truth = {0, 0, 1, 0, 0, 0};

    ASSERT_TRUE(results.size()==ground_truth.size());

    for (size_t i=0; i<results.size(); ++i)
    {
        ASSERT_EQ(ground_truth[i], results[i]);
    }

    const FLOAT_TYPE gt_cost = 18.7f;
    ASSERT_FLOAT_EQ(min_cost, gt_cost);
}
