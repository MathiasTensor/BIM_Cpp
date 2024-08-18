cmake_minimum_required(VERSION 3.27)
project(BIM_GitHub)

set(CMAKE_CXX_STANDARD 23)

set(VCPKG_POLICY_SKIP_DUMPBIN_CHECKS enabled)
set(VCPKG_POLICY_SKIP_STATIC_CHECKS enabled)
set(VCPKG_POLICY_DLLS_WITHOUT_LIBS enabled)

find_package(Boost COMPONENTS program_options REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(GSL REQUIRED)
find_package(Armadillo CONFIG REQUIRED)
find_package(FFTW3 CONFIG REQUIRED)

# Find libraries
find_library(PYTHON_LIBRARY NAMES python3.12 PATHS /Library/Frameworks/Python.framework/Versions/3.12/lib)
find_package(Matplot++ CONFIG REQUIRED)

add_executable(BIM_GitHub main.cpp
        BIM.hpp
        Boundary.hpp
        KernelIntegrationStrategy.hpp
        QuantumAnalysisTools.hpp
        Plotting.hpp
)

# Set compiler flags for GCC/Clang to avoid unsafe math optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(BIM_GitHub PRIVATE -O3 -march=native -ffp-contract=off -fno-fast-math -DEIGEN_NO_DEBUG)
endif()

# Set compiler flags for Intel compiler to ensure accurate math
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    target_compile_options(BIM_GitHub PRIVATE -O3 -fp-model precise -DEIGEN_NO_DEBUG)
endif()

#Print found packages
if (Eigen3_FOUND)
    message(STATUS "Found Eigen3")
else()
    message(FATAL_ERROR "Eigen3 not found")
endif()

if (GSL_FOUND)
    message(STATUS "Found GSL")
else()
    message(FATAL_ERROR "GSL not found")
endif()

if (Boost_FOUND)
    message(STATUS "Found Boost")
else()
    message(FATAL_ERROR "Boost not found")
endif()

if (Armadillo_FOUND)
    message(STATUS "Found Armadillo")
else()
    message(FATAL_ERROR "Armadillo not found")
endif()

if (FFTW3_FOUND)
    message(STATUS "Found FFTW3")
else()
    message(FATAL_ERROR "FFTW3 not found")
endif()

if (Matplot++_FOUND)
    message(STATUS "Found Matplot++")
else()
    message(FATAL_ERROR "Matplot++ not found")
endif()

target_link_libraries(BIM_GitHub
        PRIVATE
        Eigen3::Eigen
        GSL::gsl
        Boost::program_options
        Matplot++::cimg
        Matplot++::matplot
        armadillo
)
