cmake_minimum_required(VERSION 3.5)

project( DisplayImage )

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

add_executable(main 
    main.cpp 
    detect_and_track.cpp 
    pre_filter.cpp
    utils.cpp
)

target_link_libraries(main ${OpenCV_LIBS} Eigen3::Eigen) 