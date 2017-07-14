#include "Node.h"
#include "SnapcutTerms.h"

using namespace ROAM;

//-----------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------- LABELSPACE ----------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
LabelSpace::LabelSpace(const uint side_length)
// -----------------------------------------------------------------------------------
{
    this->fillCoordinates(side_length);
}

// -----------------------------------------------------------------------------------
void LabelSpace::Rebuild(const uint side_length)
// -----------------------------------------------------------------------------------
{
    this->indexed_coordinates.clear();
    this->fillCoordinates(side_length);
}

// -----------------------------------------------------------------------------------
uint LabelSpace::GetNumLabels() const
// -----------------------------------------------------------------------------------
{
    return static_cast<uint>(indexed_coordinates.size());
}

// -----------------------------------------------------------------------------------
void LabelSpace::fillCoordinates(const uint side_length)
// -----------------------------------------------------------------------------------
{
    int c = side_length / 2;
    int x, y;
    x = y = c;

    //inwards to outwards
    for(int levl=1; c+levl <= static_cast<int>(side_length); levl++)
    {
        for(; y<=c+levl && y < static_cast<int>(side_length); y++) // go right
            indexed_coordinates.push_back( cv::Point(x-c, y-c) );

        if (x == 0 && y == static_cast<int>(side_length)) // we are done
            break;

        for(x++,y--; x<=c+levl && x < static_cast<int>(side_length); x++)  // go down
            indexed_coordinates.push_back( cv::Point(x-c, y-c) );

        for(x--,y--; y>=c-levl ; y--)    // go left
            indexed_coordinates.push_back( cv::Point(x-c, y-c) );

        for(x--,y++; x>=c-levl ;x--)     // go up
            indexed_coordinates.push_back( cv::Point(x-c, y-c) );

        x++;
        y++;
    }
}

// -----------------------------------------------------------------------------------
cv::Point LabelSpace::GetDisplacementsFromLabel(const label l) const
// -----------------------------------------------------------------------------------
{
    assert(l < indexed_coordinates.size());
    return indexed_coordinates[l];
}

//-----------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------- NODE -------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
Node::Node(const cv::Point coordinates, const Node::Params& parameters )
// -----------------------------------------------------------------------------------
{
    node_coordinates = coordinates;
    params = parameters;
    label_space.Rebuild(parameters.label_space_side);
    remove = false;
}

// -----------------------------------------------------------------------------------
void Node::SetCoordinates(const label l)
// -----------------------------------------------------------------------------------
{
    this->node_coordinates = this->getDisplacedPointFromLabel(l);
}

// -----------------------------------------------------------------------------------
void Node::SetCoordinates(const cv::Point coordinates)
// -----------------------------------------------------------------------------------
{
    node_coordinates = coordinates;
}

// -----------------------------------------------------------------------------------
cv::Point Node::GetCoordinates() const
// -----------------------------------------------------------------------------------
{
    return node_coordinates;
}

// -----------------------------------------------------------------------------------
void Node::AddUnaryTerm(const cv::Ptr<UnaryTerm> &unary_term)
// -----------------------------------------------------------------------------------
{
    unary_terms.push_back( unary_term );
}

// -----------------------------------------------------------------------------------
void Node::SetUnary(cv::Ptr<UnaryTerm> unary_term, const size_t index)
// -----------------------------------------------------------------------------------
{
    assert(index<this->unary_terms.size());
    this->unary_terms[index] = unary_term;
}

// -----------------------------------------------------------------------------------
cv::Ptr<UnaryTerm>& Node::GetUnary(const size_t index)
// -----------------------------------------------------------------------------------
{
    assert(index<this->unary_terms.size());

    return this->unary_terms[index];
}

// -----------------------------------------------------------------------------------
void Node::ClearUnaryTerms()
// -----------------------------------------------------------------------------------
{
    unary_terms.clear();
}

// -----------------------------------------------------------------------------------
void Node::AddPairwiseTerm(const cv::Ptr<PairwiseTerm> &pairwise_term)
// -----------------------------------------------------------------------------------
{
    pairwise_terms.push_back(pairwise_term);
}

// -----------------------------------------------------------------------------------
void Node::ClearPairwiseTerms()
// -----------------------------------------------------------------------------------
{
    pairwise_terms.clear();
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE Node::GetTotalPairwiseCost(const cv::Point coordinates_a, const cv::Point coordinates_b) const
// -----------------------------------------------------------------------------------
{
    FLOAT_TYPE acum = 0.0;

    for (uint pairwise_index = 0; pairwise_index < pairwise_terms.size(); ++pairwise_index)
        acum += pairwise_terms[pairwise_index]->GetCost(coordinates_a, coordinates_b) * pairwise_terms[pairwise_index]->GetWeight();

    return acum;
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE Node::GetTotalUnaryCost(const cv::Point coordinates) const
// -----------------------------------------------------------------------------------
{
    FLOAT_TYPE acum = 0.0;

    for(uint unary_index = 0; unary_index < unary_terms.size(); ++unary_index)
        acum += unary_terms[unary_index]->GetCost(coordinates) * unary_terms[unary_index]->GetWeight();

    return acum;
}

// -----------------------------------------------------------------------------------
unsigned int Node::GetLabelSpaceSize() const
// -----------------------------------------------------------------------------------
{
    return label_space.GetNumLabels();
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE Node::GetTotalUnaryCost(const label l) const
// -----------------------------------------------------------------------------------
{
    return GetTotalUnaryCost(this->getDisplacedPointFromLabel(l));
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE Node::GetTotalPairwiseCost(const label label_a, const label label_b, const Node &node_b) const
// -----------------------------------------------------------------------------------
{
    if (pairwise_costs.size() > 0)
        return pairwise_costs[label_a*node_b.GetLabelSpaceSize() + label_b];
    else
        return GetTotalPairwiseCost(this->getDisplacedPointFromLabel(label_a),
                                node_b.getDisplacedPointFromLabel(label_b));
}

// -----------------------------------------------------------------------------------
size_t Node::GetPairwiseTermsSize() const
// -----------------------------------------------------------------------------------
{
    return pairwise_terms.size();
}

// -----------------------------------------------------------------------------------
size_t Node::GetUnaryTermsSize() const
// -----------------------------------------------------------------------------------
{
    return unary_terms.size();
}

// -----------------------------------------------------------------------------------
cv::Point Node::getDisplacedPointFromLabel(const ROAM::label l) const
// -----------------------------------------------------------------------------------
{
    return this->node_coordinates + label_space.GetDisplacementsFromLabel(l);
}


