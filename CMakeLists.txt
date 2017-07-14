cmake_minimum_required (VERSION 2.8.11)

#--------------------------------------------------------------------------
# must go before the project command
#--------------------------------------------------------------------------
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
if(DEFINED CMAKE_BUILD_TYPE)
  set_property( CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES} )
endif()

#--------------------------------------------------------------------------
# project name
#--------------------------------------------------------------------------
project(ROAM)

#--------------------------------------------------------------------------
# options
#--------------------------------------------------------------------------
option(WITH_CUDA "Use Cuda?" OFF)
option(WITH_OPENMP "Enable OpenMP support?" ON)
option(WITH_TEST "Compile tests? (requires gtest)" OFF)
option(WITH_DOCUMENTATION "Create documentation? (requires doxygen)" OFF)

#--------------------------------------------------------------------------
# cmake
#--------------------------------------------------------------------------
# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
set(CMAKE_CXX_STANDARD 14)

#--------------------------------------------------------------------------
# overcome issues caused by our friends from MS
#--------------------------------------------------------------------------
if(MSVC)
  set(CMAKE_USE_RELATIVE_PATHS ON CACHE INTERNAL "" FORCE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP") # parallel build
  add_definitions(-D_CRT_SECURE_NO_WARNINGS) # blah
  add_definitions(-DUSING_CMAKE=1)
endif()

#--------------------------------------------------------------------------
# Save libs and executables in the same place
#--------------------------------------------------------------------------
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Output directory for applications" )

#--------------------------------------------------------------------------
# add packages
#--------------------------------------------------------------------------
macro(CANONIFY_BOOL var)
  if (${var})
    set(${var} TRUE)
  else()
    set(${var} FALSE)
  endif()
endmacro()

# openmp
IF(WITH_OPENMP)
  FIND_PACKAGE(OpenMP)
  CANONIFY_BOOL(OPENMP_FOUND)
  message(STATUS "OpenMP found: " ${OPENMP_FOUND})
  IF(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  ENDIF()
  ADD_DEFINITIONS(-DWITH_OPENMP)
ENDIF()

# opencv
find_package(OpenCV REQUIRED)
CANONIFY_BOOL(OpenCV_FOUND)
message(STATUS "OpenCV found: " ${OpenCV_FOUND})

# cuda
if(WITH_CUDA)
    find_package(CUDA REQUIRED)
    CANONIFY_BOOL(CUDA_FOUND)
    message(STATUS "CUDA found: " ${CUDA_FOUND})
    if (CUDA_FOUND)
    	message("CUDA FOUND!")
    	set(CUDA_CHECK ON)
	endif()
endif()

# check if Doxygen is installed
if(WITH_DOCUMENTATION)
  find_package(Doxygen)
  CANONIFY_BOOL(DOXYGEN_FOUND)
  message(STATUS "Doxygen found: " ${DOXYGEN_FOUND})
  if (DOXYGEN_FOUND)
      # set input and output files
      set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
      set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

      # request to configure the file
      configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
      message("Doxygen build started")

      # note the option ALL which allows to build the docs together with the application
      add_custom_target( doc_doxygen ALL
          COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          COMMENT "Generating API documentation with Doxygen"
          VERBATIM )
  else (DOXYGEN_FOUND)
    message("Doxygen need to be installed to generate the doxygen documentation")
  endif (DOXYGEN_FOUND)
endif()

#--------------------------------------------------------------------------
# projects
#--------------------------------------------------------------------------
add_subdirectory(roam)
add_subdirectory(roam_apps)
add_subdirectory(tools)

