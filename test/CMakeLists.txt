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

add_executable(analytic_3d src/analytic_3d.cpp src/common.hpp)
target_link_libraries(analytic_3d PUBLIC ${ARMADILLO_LIB} ${GGM_INVERSION_LIB})

add_executable(l2_adam_5d src/l2_adam_5d.cpp src/common.hpp)
target_link_libraries(l2_adam_5d PUBLIC ${ARMADILLO_LIB} ${GGM_INVERSION_LIB})

add_executable(l2_gd_3d src/l2_gd_3d.cpp src/common.hpp)
target_link_libraries(l2_gd_3d PUBLIC ${ARMADILLO_LIB} ${GGM_INVERSION_LIB})

add_executable(l2_gd_5d src/l2_gd_5d.cpp src/common.hpp)
target_link_libraries(l2_gd_5d PUBLIC ${ARMADILLO_LIB} ${GGM_INVERSION_LIB})

add_executable(l2_optim_5d src/l2_optim_5d.cpp src/common.hpp)
target_link_libraries(l2_optim_5d PUBLIC ${ARMADILLO_LIB} ${GGM_INVERSION_LIB})

add_executable(root_find_newton_5d src/root_find_newton_5d.cpp src/common.hpp)
target_link_libraries(root_find_newton_5d PUBLIC ${ARMADILLO_LIB} ${GGM_INVERSION_LIB})

# If want to include install target
# install(TARGETS bmla_layer_1 RUNTIME DESTINATION bin)
