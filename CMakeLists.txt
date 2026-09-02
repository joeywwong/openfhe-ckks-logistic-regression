cmake_minimum_required(VERSION 3.5.1)

project(openfhe_ckks_logistic_regression_lab CXX)

# CMake learned the CXX_STANDARD value 17 in version 3.8. Keep the
# template's older CMake compatibility while still requiring a C++17-capable
# GNU or Clang compiler in the documented Linux/WSL environment.
if(CMAKE_VERSION VERSION_LESS 3.8)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17")
    else()
        message(FATAL_ERROR "CMake versions older than 3.8 require a GNU or Clang C++17 compiler")
    endif()
else()
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS ON)
endif()
option(BUILD_STATIC "Set to ON to include static versions of the library" OFF)

# The lab comparison was implemented and measured with the same OpenFHE
# installation used by the template project.
find_package(OpenFHE 1.1.2 EXACT CONFIG REQUIRED)
if(OpenFHE_FOUND)
    message(STATUS "FOUND PACKAGE OpenFHE")
    message(STATUS "OpenFHE Version: ${BASE_OPENFHE_VERSION}")
    message(STATUS "OpenFHE installed as shared libraries: ${OpenFHE_SHARED}")
    message(STATUS "OpenFHE include files location: ${OpenFHE_INCLUDE}")
    message(STATUS "OpenFHE lib files location: ${OpenFHE_LIBDIR}")
    message(STATUS "OpenFHE Native Backend size: ${OpenFHE_NATIVE_SIZE}")
else()
    message(FATAL_ERROR "PACKAGE OpenFHE NOT FOUND")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenFHE_CXX_FLAGS}")

include_directories(${OPENMP_INCLUDES})
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${OpenFHE_LIBDIR})
link_directories(${OPENMP_LIBRARIES})

if(BUILD_STATIC)
    set(CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
    link_libraries(${OpenFHE_STATIC_LIBRARIES})
else()
    set(CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS})
    if(UNIX AND NOT APPLE)
        # OpenFHE 1.1.2's pke shared library depends on binfhe. DT_RPATH is
        # transitive, unlike GNU ld's default DT_RUNPATH.
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")
    endif()
    set(CMAKE_BUILD_RPATH "${OpenFHE_LIBDIR}")
    link_libraries(${OpenFHE_SHARED_LIBRARIES})
endif()

add_library(
    lab_ml
    src/dataset.cpp
    src/logistic_regression.cpp
    src/sample_packing.cpp
)

add_library(lab_fhe src/ckks_logistic_regression.cpp)
target_link_libraries(lab_fhe PUBLIC lab_ml)

add_executable(openfhe_lab_compare app/main.cpp)
target_link_libraries(openfhe_lab_compare PRIVATE lab_fhe)
target_compile_definitions(
    openfhe_lab_compare
    PRIVATE OPENFHE_LAB_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

include(CTest)
if(BUILD_TESTING)
    find_package(PythonInterp 3 QUIET)
    if(PYTHONINTERP_FOUND)
        add_test(
            NAME gd_nag_summary_tests
            COMMAND ${PYTHON_EXECUTABLE} -m unittest discover
                -s ${CMAKE_CURRENT_SOURCE_DIR}/tests
                -p test_summarize_gd_nag.py
        )
        set_tests_properties(
            gd_nag_summary_tests
            PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()

    add_executable(plaintext_tests tests/plaintext_tests.cpp)
    target_link_libraries(plaintext_tests PRIVATE lab_ml)
    add_test(
        NAME plaintext_tests
        COMMAND plaintext_tests
            ${CMAKE_CURRENT_SOURCE_DIR}/data/LogReg_sample_dataset.csv
            ${CMAKE_CURRENT_SOURCE_DIR}/data/framingham.csv
    )
    add_executable(packing_tests tests/packing_tests.cpp)
    target_link_libraries(packing_tests PRIVATE lab_ml)
    add_test(NAME packing_tests COMMAND packing_tests
        ${CMAKE_CURRENT_SOURCE_DIR}/data/LogReg_sample_dataset.csv
        ${CMAKE_CURRENT_SOURCE_DIR}/data/framingham.csv)

    add_executable(encrypted_packing_tests tests/encrypted_packing_tests.cpp)
    target_link_libraries(encrypted_packing_tests PRIVATE lab_fhe)
    add_test(NAME encrypted_packing_tests COMMAND encrypted_packing_tests
        ${CMAKE_CURRENT_SOURCE_DIR}/data/LogReg_sample_dataset.csv
        ${CMAKE_CURRENT_SOURCE_DIR}/data/framingham.csv chebyshev)
    add_test(NAME encrypted_packing_cubic_tests COMMAND encrypted_packing_tests
        ${CMAKE_CURRENT_SOURCE_DIR}/data/LogReg_sample_dataset.csv
        ${CMAKE_CURRENT_SOURCE_DIR}/data/framingham.csv cubic)
    set_tests_properties(encrypted_packing_tests encrypted_packing_cubic_tests
        PROPERTIES LABELS "encrypted;integration" TIMEOUT 300)
endif()
