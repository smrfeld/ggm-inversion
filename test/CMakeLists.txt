cmake_minimum_required(VERSION 3.1)

set(CMAKE_OSX_DEPLOYMENT_TARGET "10.9" CACHE STRING "Minimum OS X deployment version")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(ggm_inversion_test)

include_directories(/usr/local/include)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/../bin)

find_library(ARMADILLO_LIB armadillo HINTS /usr/local/lib/)
find_library(GGM_INVERSION_LIB ggm_inversion HINTS /usr/local/lib/)

add_executable(test_3d_sgd src/test_3d_sgd.cpp src/common.hpp)
target_link_libraries(test_3d_sgd PUBLIC ${ARMADILLO_LIB} ${GGM_INVERSION_LIB})

add_executable(test_5d_sgd src/test_5d_sgd.cpp src/common.hpp)
target_link_libraries(test_5d_sgd PUBLIC ${ARMADILLO_LIB} ${GGM_INVERSION_LIB})

add_executable(test_5d_adam src/test_5d_adam.cpp src/common.hpp)
target_link_libraries(test_5d_adam PUBLIC ${ARMADILLO_LIB} ${GGM_INVERSION_LIB})

add_executable(test_5d_l_bfgs src/test_5d_l_bfgs.cpp src/common.hpp)
target_link_libraries(test_5d_l_bfgs PUBLIC ${ARMADILLO_LIB} ${GGM_INVERSION_LIB})

# If want to include install target
# install(TARGETS bmla_layer_1 RUNTIME DESTINATION bin)
