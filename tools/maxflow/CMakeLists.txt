cmake_minimum_required (VERSION 2.6)
project(bk_maxflow)

set(BKMAXFLOW_source_files
  maxflow/graph.cpp
  maxflow/maxflow.cpp
)

include_directories(maxflow)

add_library (bk_maxflow ${CC_LIBRARY_BUILDING_MODE} ${BKMAXFLOW_source_files})
set_target_properties(bk_maxflow PROPERTIES DEBUG_POSTFIX _d)