cmake_minimum_required(VERSION 3.10.2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(fly_mission)

add_executable(main
    file_parser.cpp
    drone_controller.cpp
    main.cpp
)

find_package(MAVSDK REQUIRED)

target_link_libraries(main
    MAVSDK::mavsdk
)

if(NOT MSVC)
    add_compile_options(main PRIVATE -Wall -Wextra)
else()
    add_compile_options(main PRIVATE -WX -W2)
endif()
