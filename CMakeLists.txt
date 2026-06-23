cmake_minimum_required(VERSION 3.21)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "" FORCE)
project(cuda-gemm LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED COMPONENTS cublas)

# ---- Auto-detect GPU compute capability ----
option(CUDA_GEMM_AUTO_DETECT_SM "Auto-detect SM arch" ON)
set(_arch_needs_detection OFF)
if(CUDA_GEMM_AUTO_DETECT_SM)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL ""
       OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
        set(_arch_needs_detection ON)
    endif()
endif()
if(CUDA_GEMM_AUTO_DETECT_SM AND _arch_needs_detection)
    set(_detected_arch "")
    find_program(NVIDIA_SMI_EXECUTABLE nvidia-smi)
    if(NVIDIA_SMI_EXECUTABLE)
        execute_process(
            COMMAND ${NVIDIA_SMI_EXECUTABLE} --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE _smi_output RESULT_VARIABLE _smi_result
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(_smi_result EQUAL 0 AND NOT _smi_output STREQUAL "")
            string(REPLACE " " "" _smi_output "${_smi_output}")
            string(REPLACE "\r" "" _smi_output "${_smi_output}")
            string(REPLACE "\n" ";" _cap_list "${_smi_output}")
            set(_cap_values "")
            foreach(_cap IN LISTS _cap_list)
                if(_cap MATCHES "^[0-9]+\\.[0-9]+$")
                    string(REPLACE "." "" _cap_digits "${_cap}")
                    list(APPEND _cap_values "${_cap_digits}")
                endif()
            endforeach()
            if(_cap_values)
                list(SORT _cap_values COMPARE NATURAL ORDER DESCENDING)
                list(GET _cap_values 0 _detected_arch)
            endif()
        endif()
    endif()
    if(_detected_arch)
        message(STATUS "Detected GPU compute capability: sm_${_detected_arch}")
        set(CMAKE_CUDA_ARCHITECTURES "${_detected_arch}" CACHE STRING "" FORCE)
    else()
        set(_default_arch "90")
        message(WARNING "Cannot detect GPU; defaulting to sm_${_default_arch}")
        set(CMAKE_CUDA_ARCHITECTURES "${_default_arch}" CACHE STRING "" FORCE)
    endif()
endif()

# ---- Explicit source list (avoids GLOB_RECURSE picking up build artifacts) ----
set(CUDA_GEMM_SOURCES
    main.cu
    sgemm/naive_sgemm.cu
    sgemm/v1_sgemm.cu
    sgemm/v2_sgemm.cu
    sgemm/v3_sgemm.cu
    hgemm/v1_hgemm.cu
    hgemm/v2_hgemm.cu
    hgemm/v3_hgemm.cu
    hgemm/v4_hgemm.cu
)

add_executable(cuda_gemm ${CUDA_GEMM_SOURCES})

target_compile_options(cuda_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O2>)
target_link_libraries(cuda_gemm PRIVATE CUDA::cublas CUDA::cudart)

# Convenience targets
add_custom_target(run
    COMMAND cuda_gemm
    DEPENDS cuda_gemm
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(bench
    COMMAND cuda_gemm --bench
    DEPENDS cuda_gemm
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
