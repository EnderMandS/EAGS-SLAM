cmake_minimum_required(VERSION 3.4)
set(PROJECT_NAME "REVO")
project(${PROJECT_NAME})

SET(CMAKE_BUILD_TYPE RELEASE)

# Set the output directories relative to the VO directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/lib)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -O1 -msse3 -msse4.1 -mavx2 -fopenmp")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O1 -msse3 -msse4.1 -mavx2 -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -msse3 -msse4.1 -mavx2 -fopenmp")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -msse3 -msse4.1 -mavx2 -fopenmp")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
if(COMPILER_SUPPORTS_CXX17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has c++17 support.")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} not support c++17.")
endif()

# Initialize variables to hold all source and header files
set(FILE_DIRS system utils datastructures io)
set(ALL_SOURCES "main.cpp")
set(ALL_HEADERS "")

# OpenCV
find_package(OpenCV 3.4 REQUIRED PATHS /opt/opencv/34)

# Boost
find_package(Boost COMPONENTS system filesystem REQUIRED)

# PyBind
set(CONDA_PREFIX "~/.conda/envs/eags")
message("CONDA_PREFIX: ${CONDA_PREFIX}")
find_package(pybind11 REQUIRED PATHS ${CONDA_PREFIX}/lib/python3.10/site-packages/pybind11)
include_directories(${CONDA_PREFIX}/include/python3.10)
include_directories(${CONDA_PREFIX}/lib/python3.10/site-packages/pybind11/include)
link_directories(${CONDA_PREFIX}/lib)

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

# Sophus
include_directories("thirdparty/Sophus")

# Iterate over each directory in FILE_DIRS
message("FILE_DIRS: ${FILE_DIRS}")
foreach(DIR ${FILE_DIRS})
    # Collect all source files in the src directory
    file(GLOB_RECURSE SOURCES "${DIR}/*.cpp" "${DIR}/*.cc")
    list(APPEND ALL_SOURCES ${SOURCES})

    # Collect all header files in the include directory
    file(GLOB_RECURSE HEADERS "${DIR}/*.h" "${DIR}/*.hpp")
    list(APPEND ALL_HEADERS ${HEADERS})

    # Add include directories
    include_directories(${DIR})
endforeach()

# add_executable(${PROJECT_NAME} ${ALL_SOURCES} ${ALL_HEADERS})
# target_link_libraries(${PROJECT_NAME}
#     ${LIBS}
#     ${OpenCV_LIBS} 
#     ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY}
#     ${CUDA_LIBRARIES} ${CUDNN_LIBRARIES} ${TENSORRT_LIBRARIES}
# )

# Generate library for python
add_library(VisualOdom MODULE pybind/vo.cpp ${ALL_SOURCES})
target_link_libraries(VisualOdom PRIVATE
    ${LIBS}
    ${OpenCV_LIBS}
    ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY}
    # ${CUDA_LIBRARIES} ${CUDNN_LIBRARIES} ${TENSORRT_LIBRARIES}
    pybind11::module
)
set_target_properties(VisualOdom 
    PROPERTIES
    PREFIX ""
    SUFFIX ".so")
