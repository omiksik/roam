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

#include <vector>

#include <opencv2/core.hpp>

#include "EnergyTerms.h"
#include "GreenTheorem.h"

namespace ROAM
{

typedef unsigned short label;

/*!
 * \brief The LabelSpace struct
 */
 // -----------------------------------------------------------------------------------
struct LabelSpace
// -----------------------------------------------------------------------------------
{
    explicit LabelSpace(const uint side_length = 3);

    void Rebuild(const uint side_length);

    cv::Point GetDisplacementsFromLabel(const ROAM::label l) const;
    uint GetNumLabels() const;

protected:
    void fillCoordinates(const uint side_length);

    std::vector<cv::Point> indexed_coordinates;
};


/*!
 * \brief The Node class
 */
// -----------------------------------------------------------------------------------
class Node
// -----------------------------------------------------------------------------------
{
public:
    /*!
     * \brief The Params struct
     */
    struct Params
    {
        // ---------------------------------------------------------------------------
        explicit Params(const unsigned int label_space_side = 5)
        // ---------------------------------------------------------------------------
        {
            this->label_space_side = label_space_side;
        }

        unsigned int label_space_side;
    };

    explicit Node(const cv::Point coordinates, const Node::Params& parameters = Node::Params());

    void SetCoordinates(const label l);
    void SetCoordinates(const cv::Point coordinates);
    cv::Point GetCoordinates() const;

    void AddUnaryTerm(const cv::Ptr<UnaryTerm>& unary_term);
    void SetUnary(cv::Ptr<UnaryTerm> unary_term, const size_t index=0);
    cv::Ptr<UnaryTerm> &GetUnary(const size_t index=0);
    void ClearUnaryTerms();

    void AddPairwiseTerm(const cv::Ptr<PairwiseTerm>& pairwise_term);
    void ClearPairwiseTerms();

    /*!
     * \brief GetTotalUnaryCost returns the sum of all active unary costs for this node
     * \param coordinates of the point possible move
     * \return
     */
    FLOAT_TYPE GetTotalUnaryCost(const cv::Point coordinates) const;

    /*!
     * \brief GetTotalPairwiseCost returns the sum of all the active pairwise costs for this node
     * \param coordinates_a
     * \param coordinates_b
     * \return
     */
    FLOAT_TYPE GetTotalPairwiseCost(const cv::Point coordinates_a, const cv::Point coordinates_b) const;


    /*!
     * \brief GetTotalUnaryCost returns the total unary cost for this node given a label
     * \param label
     * \return
     */
    FLOAT_TYPE GetTotalUnaryCost(const label l) const;

    /*!
     * \brief GetTotalPairwiseCost returns the total unary cost for this node given pair of labels
     * \param label_a
     * \param label_b
     * \param node_b is also needed here to be able to search in its own label_space
     * \return
     */
    FLOAT_TYPE GetTotalPairwiseCost(const label label_a, const label label_b, const Node& node_b) const;


    size_t GetPairwiseTermsSize() const;
    size_t GetUnaryTermsSize() const;

    /*!
     * \brief ComputePairwiseCostsCuda
     * \param node_b
     */
    void ComputePairwiseCostsCuda(const Node& node_b);

    /*!
     * \brief GetLabelSpaceSize
     * \return
     */
    unsigned int GetLabelSpaceSize() const;

public:
    std::vector< cv::Ptr<UnaryTerm> > unary_terms;
    std::vector< cv::Ptr<PairwiseTerm> > pairwise_terms;
    bool remove;
    cv::Point getDisplacedPointFromLabel(const label l) const;

protected:
    LabelSpace label_space;
    Params params;
    cv::Point node_coordinates;
    std::vector<FLOAT_TYPE> pairwise_costs;
};


}// namespace roam
