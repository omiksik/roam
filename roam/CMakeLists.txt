cmake_minimum_required (VERSION 2.8.11)
project(roam_lib)

# Find the OpenCV library
find_package(OpenCV REQUIRED)

#--------------------------------------------------------------------------
# check if OpenCV contrib exists
#--------------------------------------------------------------------------
include(CheckIncludeFileCXX)
check_include_file_cxx(opencv2/ximgproc.hpp HAVE_OPENCV_CONTRIB)
if(HAVE_OPENCV_CONTRIB)
    add_definitions(-DHAVE_OPENCV_CONTRIB)
endif(HAVE_OPENCV_CONTRIB)

#--------------------------------------------------------------------------
# ROAM
#--------------------------------------------------------------------------

set(roam_SRC
  src/EnergyTerms.cpp
  src/SnapcutTerms.cpp
  src/GreenTheorem.cpp
  src/Node.cpp
  src/ClosedContour.cpp
  src/StarGraph.cpp
  src/DynamicProgramming.cpp
  src/VideoSegmenter.cpp
  src/RotatedRect.cpp
  src/ContourWarper.cpp
  src/GlobalModel.cpp
  src/GC_Energy.cpp
  src/Reparametrization.cpp
  include/Configuration.h
  include/ClosedContour.h
  include/StarGraph.h
  include/EnergyTerms.h
  include/SnapcutTerms.h
  include/GreenTheorem.h
  include/DynamicProgramming.h
  include/DynamicProgramming_CPUGPU.h
  include/VideoSegmenter.h
  include/RotatedRect.h
  include/ContourWarper.h
  include/GlobalModel.h
  include/GC_Energy.h
  include/Reparametrization.h
)

# Use CUDA if Option was enabled
if(CUDA_CHECK)
   message("Compiling with CUDA")
   add_definitions(-DWITH_CUDA)

   if(NOT DEFINED CUDA_COMPUTE_CAPABILITY)
    include(${PROJECT_SOURCE_DIR}/cmake/CUDACheckCompute.cmake)

    set(CUDA_ARCH ${CUDA_COMPUTE_CAPABILITY} CACHE STRING "CUDA architectures to compile for")
    string(REPLACE ";" " " CUDA_ARCH_LIST ${CUDA_ARCH})
    foreach(a ${CUDA_ARCH_LIST})
    set(CUDA_ARCH_OPTIONS "${CUDA_ARCH_OPTIONS} -gencode arch=compute_${a},code=compute_${a}")
    endforeach()
   endif()

   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")
   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; "${CUDA_ARCH_OPTIONS} --use_fast_math -std=c++11 -D_MWAITXINTRIN_H_INCLUDED")

   set(CUDASRCS cuda/roam_cuda.cu cuda/roam_cuda.h cuda/dp_cuda.h cuda/dp_cuda.cu)
   cuda_add_library(roam_gpu ${CUDASRCS} OPTIONS "${CUDA_ARCH_OPTIONS} --use_fast_math")

   # Includes
   include_directories(include ${OpenCV_INCLUDE_DIRS} cuda ${TOOLS_include_dir})

   add_library(roam_lib ${roam_SRC})
   target_link_libraries(roam_lib roam_gpu ${OpenCV_LIBS} cv_tools_lib om_utils bk_maxflow)
   set_target_properties(roam_lib PROPERTIES DEBUG_POSTFIX _d)

else()
    message("Compiling without CUDA")
    # Includes
    include_directories(include ${OpenCV_INCLUDE_DIRS} ${TOOLS_include_dir})

    add_library(roam_lib ${roam_SRC})
    target_link_libraries(roam_lib ${OpenCV_LIBS} cv_tools_lib om_utils bk_maxflow)
    set_target_properties(roam_lib PROPERTIES DEBUG_POSTFIX _d)

endif()

#--------------------------------------------------------------------------
# Tests (rquires gtest)
#--------------------------------------------------------------------------
if(WITH_TESTS)
  find_package(GTest)
  if (${GTEST_FOUND})
     add_subdirectory(test)
  endif()
endif()

