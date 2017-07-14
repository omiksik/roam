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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "../include/Configuration.h"
#include "../include/DynamicProgramming.h"

#include "omp.h"
#include "cuda_profiler_api.h"

#ifdef __CUDACC__
#define CUDA_METHOD __host__ __device__
#else
#define CUDA_METHOD
#endif

#include "math.h"

namespace cuda_roam
{

// -------------------------------------------------------------------------------
FLOAT_TYPE CudaDPMinimize(const ROAM::DPTableUnaries& unary_costs,
                          const ROAM::DPTablePairwises& pairwise_costs,
                          std::vector<ROAM::label> &path);
// -------------------------------------------------------------------------------

} // namespace cuda_roam

