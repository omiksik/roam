cmake_minimum_required (VERSION 2.6)
project(om_utils)

set(UTILS_source_files
  utils/timer.cpp
  utils/file_utils.cpp
  utils/color_map.cpp
  include/roam/utils/confusion_matrix.h
)

include_directories(include include/roam include/roam/utils)


add_library (om_utils ${CC_LIBRARY_BUILDING_MODE} ${UTILS_source_files})
set_target_properties(om_utils PROPERTIES DEBUG_POSTFIX _d)
