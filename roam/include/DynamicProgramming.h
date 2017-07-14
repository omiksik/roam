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
#include <memory>
#include <limits>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdint.h>

#include "Configuration.h"


namespace ROAM
{

typedef unsigned short label;
typedef std::vector< std::vector< FLOAT_TYPE > > DPTableUnaries;
typedef std::vector< std::vector< FLOAT_TYPE > > DPTableNodePairwises;
typedef std::vector< DPTableNodePairwises > DPTablePairwises;
typedef std::vector< DPTablePairwises > DPTableStarPairwises;
typedef std::vector< DPTableUnaries > DPTableStarUnaries;
typedef std::vector< DPTableUnaries > DPVectorTableUnaries;
typedef std::vector<std::vector<std::vector<label> > > LabelTable;

}
