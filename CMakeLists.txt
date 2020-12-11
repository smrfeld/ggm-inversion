cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(ggm_inversion VERSION 0.1.0)

# Include dir
include_directories(/usr/local/include)

# Src
AUX_SOURCE_DIRECTORY(src SRC_FILES)

# Headers
set(PROJECT_SOURCE_DIR "src")
set(PROJECT_INCLUDE_DIR "include/ggm_inversion_bits")

# Source files
set(SOURCE_FILES
    ${PROJECT_INCLUDE_DIR}/optimizer_adam.hpp
    ${PROJECT_INCLUDE_DIR}/optimizer_base.hpp
    ${PROJECT_INCLUDE_DIR}/optimizer_l_bfgs.hpp
    ${PROJECT_INCLUDE_DIR}/optimizer_gd.hpp
    ${PROJECT_INCLUDE_DIR}/helpers.hpp
    ${PROJECT_INCLUDE_DIR}/optimizer_optim.hpp
    ${PROJECT_SOURCE_DIR}/optimizer_adam.cpp
    ${PROJECT_SOURCE_DIR}/optimizer_base.cpp
    ${PROJECT_SOURCE_DIR}/optimizer_l_bfgs.cpp
    ${PROJECT_SOURCE_DIR}/optimizer_gd.cpp
    ${PROJECT_SOURCE_DIR}/helpers.cpp
    ${PROJECT_SOURCE_DIR}/optimizer_optim.cpp
)

# Set up such that XCode organizes the files correctly
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES})

# Required library
find_library(ARMADILLO_LIB armadillo HINTS /usr/local/lib/ REQUIRED)
find_library(OPTIM_LIB optim HINTS /usr/local/lib/ REQUIRED)

# Add library
add_library(ggm_inversion SHARED ${SOURCE_FILES})

# Link
target_link_libraries(ggm_inversion PUBLIC ${ARMADILLO_LIB} ${OPTIM_LIB})

# Include directories
target_include_directories(ggm_inversion PRIVATE include/ggm_inversion_bits)

# Install
install(TARGETS ggm_inversion DESTINATION lib)

# Install the headers
install(FILES include/ggm_inversion DESTINATION include)

# Create base directory
install(DIRECTORY include/ggm_inversion_bits DESTINATION include)