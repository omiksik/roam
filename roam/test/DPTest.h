#include "DynamicProgramming.h"
#include "ClosedContour.h"
#include "EnergyTerms.h"
#include "Node.h"
#include "RotatedRect.h"
#include "ContourWarper.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>


TEST(DPTest, ClosedChainCompetition1)
{
    const int num_labels=2, num_nodes=3;
    std::shared_ptr<roam::ClosedChainDPTable> dp_table = std::make_shared<roam::ClosedChainDPTable>(num_labels, num_nodes);
    dp_table->Initialize();
    roam::DPTableUnaries& unary_costs = dp_table->unary_costs;
    roam::DPTablePairwises& pairwise_costs = dp_table->pairwise_costs;

    unary_costs = { {2.0f, 1.0f}, {3.0f, 0.1f}, {0.1f, 0.1f} };

    pairwise_costs = { { {2.0f,1.0f},{0.5f,3.0f} },
                       { {0.5f,0.5f},{0.5f,0.2f} },
                       { {1.0f,1.0f},{0.0f,1.0f} } };

    roam::ClosedChainDP  dp0;
    roam::ClosedChainDP1 dp1;
    roam::ClosedChainDPCuda dp2;

    FLOAT_TYPE min_cost0;
    FLOAT_TYPE min_cost1;
    FLOAT_TYPE min_cost2;

    std::vector<roam::label> results0 = dp0.Minimize(dp_table, min_cost0);
    std::vector<roam::label> results1 = dp1.Minimize(dp_table, min_cost1);
    std::vector<roam::label> results2 = dp2.Minimize(dp_table, min_cost2);
    std::vector<roam::label> ground_truth = {0,1,1};

    ASSERT_TRUE(results0.size()==ground_truth.size());
    ASSERT_TRUE(results1.size()==ground_truth.size());
    ASSERT_TRUE(results2.size()==ground_truth.size());

    std::cout<<"PATH G => ";
    for (size_t i=0; i<ground_truth.size(); ++i)
        std::cout<<ground_truth[i]<<" ";
    std::cout<<std::endl;

    std::cout<<"PATH 0 => ";
    for (size_t i=0; i<results0.size(); ++i)
        std::cout<<results0[i]<<" ";
    std::cout<<std::endl;

    std::cout<<"PATH 1 => ";
    for (size_t i=0; i<results1.size(); ++i)
        std::cout<<results1[i]<<" ";
    std::cout<<std::endl;

    std::cout<<"PATH 2 => ";
    for (size_t i=0; i<results2.size(); ++i)
        std::cout<<results2[i]<<" ";
    std::cout<<std::endl;


    for (size_t i=0; i<ground_truth.size(); ++i)
    {
        ASSERT_EQ(ground_truth[i], results0[i]);
        ASSERT_EQ(ground_truth[i], results1[i]);
        ASSERT_EQ(ground_truth[i], results2[i]);
    }

    ASSERT_EQ(min_cost0, 3.4f);
    ASSERT_EQ(min_cost1, 3.4f);
    ASSERT_EQ(min_cost2, 3.4f);
}

TEST(DPTest, StarDP)
{
    const int num_labels=2, num_nodes=3;
    std::shared_ptr<roam::StarDPTable> dp_table = std::make_shared<roam::StarDPTable>(num_labels, num_nodes);
    dp_table->Initialize();
    roam::DPTableUnaries& unary_costs = dp_table->unary_costs;
    roam::DPTablePairwises& pairwise_costs = dp_table->pairwise_costs;

    unary_costs = { {1.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f} };

    pairwise_costs = { { {1.0f,1.0f},{1.0f,0.0f} },
                       { {1.0f,1.0f},{0.0f,1.0f} } };

    roam::StarDP  dp_star;

    FLOAT_TYPE min_cost;

    std::vector<roam::label> results = dp_star.Minimize(dp_table, min_cost);
    std::vector<roam::label> ground_truth = {1,1,0};

    for (size_t i=0; i<results.size(); ++i)
        ASSERT_EQ(results[i],ground_truth[i]);
}

TEST(DPTest, TreeDP)
{
    const int num_labels=2, num_gc=2;
    //TreeDPTable(const uint16_t max_number_labels, const uint16_t number_root_children);
    std::shared_ptr<roam::TreeDPTable> dp_table = std::make_shared<roam::TreeDPTable>(num_labels, num_gc);

    dp_table->Initialize();

    //std::vector< DPTableUnaries >
    roam::DPTableStarUnaries& child_unary_costs = dp_table->children_unaries;

    //std::vector< DPTablePairwises > DPTableStarPairwises
    roam::DPTableStarPairwises& child_pairwise_costs = dp_table->children_pairwises;

    std::vector<uint16_t>& parenthood = dp_table->parent_of_children;

    roam::DPTableUnaries& unary_costs = dp_table->unary_costs;
    roam::DPTablePairwises& pairwise_costs = dp_table->pairwise_costs;

    unary_costs = { {0.0f, 1.0f}, {1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f} };

    pairwise_costs = { { {.0f,.0f},{.0f,.0f} },
                       { {.0f,.0f},{.0f,.0f} },
                       { {.0f,.0f},{.0f,.0f} } };


    roam::DPTableUnaries t1 = { {1.0f, 0.0f}, {0.0f, 0.0f} };
    roam::DPTableUnaries t2 = { {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f} };
    roam::DPTableUnaries t3 = { {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f} };

    roam::DPTablePairwises p1 = { { {.0f,1.5f},{.0f,1.5f} },
                                  { {.0f,.0f},{.0f,.0f} } };
    roam::DPTablePairwises p2 = { { {.0f,.0f},{.0f,.0f} },
                                  { {.0f,.0f},{.0f,.0f} },
                                  { {.0f,.0f},{.0f,.0f} } };
    roam::DPTablePairwises p3 = { { {.0f,.0f},{.0f,.0f} },
                                  { {.0f,.0f},{.0f,.0f} },
                                  { {.0f,.0f},{.0f,.0f} },
                                  { {.0f,.0f},{.0f,.0f} } };

    parenthood.push_back(0);
    child_unary_costs.push_back( t1 );

    parenthood.push_back(1);
    child_unary_costs.push_back( t2 );

    parenthood.push_back(2);
    child_unary_costs.push_back( t3 );

    child_pairwise_costs.push_back(p1);
    child_pairwise_costs.push_back(p2);
    child_pairwise_costs.push_back(p3);


    roam::TreeDP dp_tree;
    FLOAT_TYPE min_cost;

    std::vector<roam::label> results = dp_tree.Minimize(dp_table, min_cost);

    for (auto i:results)
        std::cerr<<" "<<i<<" ";

    std::cerr<<std::endl;
    std::cerr<<"MIN COST = "<<min_cost<<std::endl;

}
