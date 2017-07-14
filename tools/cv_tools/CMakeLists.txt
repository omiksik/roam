cmake_minimum_required (VERSION 2.6)
project(cv_tools_lib)

# Find the OpenCV library
find_package(OpenCV REQUIRED)

set(TOOLS_includes
  include/ROAM_KCF.h
  include/ROAM_GMM.h
)

set(TOOLS_source_files
  src/ROAM_KCF.cpp
)

include_directories(include
                    ${OpenCV_INCLUDE_DIRS})

add_library (cv_tools_lib ${CC_LIBRARY_BUILDING_MODE} ${TOOLS_source_files} ${TOOLS_includes})
target_link_libraries(cv_tools_lib ${OpenCV_LIBS})
set_target_properties(cv_tools_lib PROPERTIES DEBUG_POSTFIX _d)