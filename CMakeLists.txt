cmake_minimum_required (VERSION 3.5)
project (BarcodeDetector CUDA CXX)

include(FetchContent)

FetchContent_Declare(
  GoogleBenchmark
  URL https://github.com/google/benchmark/archive/v1.4.1.tar.gz
  )

FetchContent_Declare(
  CLI11
  URL https://github.com/CLIUtils/CLI11/archive/v1.8.0.tar.gz
  )

FetchContent_Declare(
  spdlog
  URL https://github.com/gabime/spdlog/archive/v1.4.2.tar.gz
  )


if (NOT GoogleBenchmark_POPULATED)
  FetchContent_Populate(GoogleBenchmark)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "From Gtest")
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "From Gtest")
  add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR})
endif()

FetchContent_MakeAvailable(spdlog)
FetchContent_MakeAvailable(CLI11)

#set(CMAKE_CXX_STANDARD LANGUAGES 11)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} )
find_package( OpenCV REQUIRED)
find_package(PNG REQUIRED)
find_package(CUDA REQUIRED)
find_package(TBB REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")


include_directories(
        src/
        src/detectors/

    )


add_executable(detector)
target_compile_features( detector PUBLIC cxx_std_14 )
target_link_libraries( detector PRIVATE ${OpenCV_LIBS}  spdlog::spdlog )


target_sources(detector PRIVATE
        src/main.hpp
        src/main.cu
        src/detectors/detectorInterface.hpp
        src/detectors/detectorInterface.cpp
        src/detectors/cpu_baseline.hpp
        src/detectors/cpu_baseline.cpp
        src/detectors/gpu_baseline.cu
        src/detectors/gpu_baseline.hpp
        )


set_target_properties(detector PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
