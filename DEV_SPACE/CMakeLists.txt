cmake_minimum_required(VERSION 3.5)

project( DisplayImage )

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

add_executable( Stitching Stitching.cpp)
add_executable( sift sift_matcher.cpp)
add_executable( sift_waypoint sift_waypoint.cpp)
add_executable( vo visual_odometry.cpp)
add_executable(ekf_vo kalman_visual_odometry.cpp)
add_executable(vo_wp visual_odometry_waypoint.cpp)
add_executable(filter_match pre_filter.cpp)
add_executable(blob blob_tracking.cpp)
add_executable(run_detect runway_detection.cpp)

target_link_libraries( run_detect ${OpenCV_LIBS})
target_link_libraries( blob ${OpenCV_LIBS})
target_link_libraries( filter_match ${OpenCV_LIBS})
target_link_libraries( vo ${OpenCV_LIBS})
target_link_libraries( sift ${OpenCV_LIBS})
target_link_libraries( sift_waypoint ${OpenCV_LIBS})
target_link_libraries( Stitching ${OpenCV_LIBS} ) 
target_link_libraries(ekf_vo ${OpenCV_LIBS} Eigen3::Eigen) 
target_link_libraries(vo_wp ${OpenCV_LIBS} Eigen3::Eigen) 