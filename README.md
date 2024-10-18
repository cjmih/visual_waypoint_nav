# visual_waypoint_nav
Visual Waypoint based navigation system developed for aerial vehicles to operate in GPS denied environments using Visual Odometry.

# Background and Purpose:
The purpose of the repo is a workspace for my masters project which is a navigation algorithm that allows UAV's to operate in GPS-deinied enviornments using a single downward facing camera. The project targets using Visual Odometry (VO) rather than Visual Inertial Odometry (VIO) to attempt to solve the problem by focusing on navigation in the 2D plane. The project works by using VO to track the drones heading over time and compares the camera stream to Google Earth satellite view to extract the drones global position. Visual waypoints/landmarks are classified in the Google Earth view and used as navigation waypoints rather than traditional GPS Latitude and Longitude Waypoints. 
![Screenshot from 2024-10-18 12-43-28](https://github.com/user-attachments/assets/0433d09b-ff3f-4f6a-976a-81e674f07872)

# Directory:
DEV_SPACE contains module prototypes and experiements with different algorithms
NAV_SPACE contains modules for drone control using mavsdk 
CAM_SPACE contains camera drivers and onboard compute kernels

# Build and compile (built on Ubuntu 22.04):
cmake .
make

# Run: 
./main <link_to_video file>

** Example usage: **
./main DJI_0189.MOV

# Expected Output: 
Two windows: 
"Runway Features" shows detected features on the frames of the input videofile
"Heading Animation" shows the extracted yaw orientation from the Visual Odometry algorithm 

# Works in Progress: 
-Fix DEV_SPACE EKF
-Dockerize application
-Make heading animation less deplorable/more sensible
-Finalize localization on global map in DEV_SPACE -> combine with main
-Tune PI-Controller in NAV_SPACE
-Upload camera drivers
-Hardmount camera and onboard compute platform to drone (currently mounted to gimbal and using post-processing) -> benchtop comms test works as expected (need to confirm signal range->purchase antennas as needed)
