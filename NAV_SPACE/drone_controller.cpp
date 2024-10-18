//MAVSDK requirements
#include <mavsdk/mavsdk.h>
#include <mavsdk/plugins/action/action.h>
#include <mavsdk/plugins/mission/mission.h>
#include <mavsdk/plugins/telemetry/telemetry.h>
#include <mavsdk/plugins/offboard/offboard.h>

//general
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <math.h>
#include <vector>
#include <sstream>
#include <iterator> 
#include <bits/stdc++.h>

// custom includes
#include "drone_controller.hpp"

using std::chrono::seconds;
using std::this_thread::sleep_for;

//defining MissionItem
mavsdk::Mission::MissionItem make_mission_item(
  double latitude_deg,
  double longitude_deg,
  float relative_altitude_m,
  float speed_m_s,
  bool is_fly_through,
  float gimbal_pitch_deg,
  float gimbal_yaw_deg,
  mavsdk::Mission::MissionItem::CameraAction camera_action)
{
  mavsdk::Mission::MissionItem new_item{};
  new_item.latitude_deg = latitude_deg;
  new_item.longitude_deg = longitude_deg;
  new_item.relative_altitude_m = relative_altitude_m;
  new_item.speed_m_s = speed_m_s;
  new_item.is_fly_through = is_fly_through;
  new_item.gimbal_pitch_deg = gimbal_pitch_deg;
  new_item.gimbal_yaw_deg = gimbal_yaw_deg;
  new_item.camera_action = camera_action;
  return new_item;
}

//connecting URL 
void usage(const std::string& bin_name)
{
  std::cerr << "Usage : " << bin_name << " <connection_url> <path_to_txt_file> <flight type>\n"
            << "Connection URL format should be :\n"
            << " For TCP : tcp://[server_host][:server_port]\n"
            << " For UDP : udp://[bind_host][:bind_port]\n"
            << " For Serial : serial:///path/to/serial/dev[:baudrate]\n"
            << "For example, to connect to the simulator use URL: udp://:14540\n"
            << "Path to file format should be: \n"
            << "../pattern_data/<file_name>.txt \n"
            << "Flight type options are: \n"
            << "For using offboard controls with points in World Reference Frame of XYZ: offboard \n"
            << "For using mission controls with points in World Reference Frame of Latitude, Longitude, Altitude: mission \n";
}

// connecting to simulator
std::shared_ptr<mavsdk::System> get_system(mavsdk::Mavsdk& mavsdk)
{
  std::cout << "Waiting to discover system...\n";
  auto prom = std::promise<std::shared_ptr<mavsdk::System>>{};
  auto fut = prom.get_future();
  mavsdk.subscribe_on_new_system([&mavsdk, &prom]() {
    auto system = mavsdk.systems().back();
      if (system->has_autopilot()) {
        std::cout << "Discovered autopilot\n";
        mavsdk.subscribe_on_new_system(nullptr);
        prom.set_value(system);
    }
  });
  //timeout if no connection
  if (fut.wait_for(seconds(5)) == std::future_status::timeout) {
    std::cerr << "No autopilot found.\n";
    return {};
  }

  // Get discovered system now.
  return fut.get();
}

//arm drone
int arm_drone(mavsdk::Action& action){
  std::cout << "Arming...\n";
    const mavsdk::Action::Result arm_result = action.arm();
    if (arm_result != mavsdk::Action::Result::Success) {
      std::cerr << "Arming failed: " << arm_result << '\n';
      return 1;
    }
    std::cout << "Armed.\n";
  return 0;
}

//function that takes a map of waypoints and adds it to a vector of MissionItem's
std::vector<mavsdk::Mission::MissionItem> setup_mission(std::map<int, std::tuple<double, double, double>>& waypoints){  
  //define output vector
  std::vector<mavsdk::Mission::MissionItem> mission_items;
  //iterate over input waypoints
  //sorts the waypoints in order of ID# 
  for (auto i = waypoints.begin(); i != waypoints.end(); i++){
    //define a customizable mission_item which only takes the XYZ coordinates from our waypoint, rest of values are set to standard value
    mavsdk::Mission::MissionItem mission_item= make_mission_item(
      std::get<0>(i->second),
      std::get<1>(i->second),
      std::get<2>(i->second),
      5.0f,
      false,
      0.0f,
      0.0f,
      mavsdk::Mission::MissionItem::CameraAction::None
      );
    //add mission item to return vector
    mission_items.push_back(mission_item);
    
    //Uncomment if you would like to see uploaded waypoints, commented out for speed
    // std::cout << "ID: " << i->first << '\n'
    //           << "Position X: " << std::get<0>(i->second) << '\n'
    //           << "Position Y: " << std::get<1>(i->second) << '\n'
    //           << "Position Z: " << std::get<2>(i->second) << '\n'
    //           << '\n';
  }
  //return the MissionItem from the .txt file
  return mission_items;
}

//custom Proportional-Integral Controller for velocity takes specific gains and error to axis
std::vector<double> velocity_pi_controller(double kp, double ki, double error, double int_error, double timestep){
  //setting output to zero
  std::vector<double> output(2); 
  //calculates the integral error term
  double integral_error_term = int_error + error*timestep;
  //calculates the output of the controller
  double u = kp*error + ki*integral_error_term;
  //store the output and current integral error
  output[0] = u;
  output[1] = integral_error_term;
  //return output of pid controller 
  return output;
}

//function that performs offboard controls on a map of waypoints
int fly_offboard(mavsdk::Offboard& offboard, mavsdk::Telemetry& telemetry, std::map<int, std::tuple<double, double, double>> flight_waypoints, mavsdk::Telemetry::PositionVelocityNed& origin){
  
  // Send it once before starting offboard, otherwise it will be rejected.
  const mavsdk::Offboard::VelocityNedYaw stay_vel{};
  offboard.set_velocity_ned(stay_vel);
  const mavsdk::Offboard::PositionNedYaw stay{};
  offboard.set_position_ned(stay);

  //confirms if offboard has started or return 1 for error
  mavsdk::Offboard::Result offboard_result = offboard.start();
  if (offboard_result != mavsdk::Offboard::Result::Success) {
    std::cerr << "Offboard start failed: " << offboard_result << '\n';
    return 1;
  }
  std::cout << "Offboard started\n";

  //iterates over input map and flies to the points
  for (auto i = flight_waypoints.begin(); i != flight_waypoints.end(); i++){
    //excracting the x,y,z position per line to fly
    double target_x_pos = std::get<0>(i->second);
    double target_y_pos = std::get<1>(i->second);
    double target_z_pos = std::get<2>(i->second);

    //getting the current position of the drone    
    auto position = telemetry.position_velocity_ned();

    //calculating current displacement
    double dx = target_x_pos - position.position.north_m;
    double dy = target_y_pos - position.position.east_m;
    double dz = -target_z_pos - position.position.down_m;

    //calculating the yaw to the next waypoint
    double target_yaw = atan2(dy, dx)*180/M_PI;
    std::cout << "Target Yaw: " << target_yaw << '\n';

    //calculating the error term for the system using euclidean distance
    double euclidean_distance = std::sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2));

    //yawing the drone in the direction of the target point so the velocity controller moves in a straight line not a curve
    mavsdk::Offboard::VelocityNedYaw turn_east{};
    turn_east.yaw_deg = target_yaw;
    turn_east.north_m_s = 0.0;
    turn_east.east_m_s = 0.0;
    turn_east.down_m_s = 0.0;
    offboard.set_velocity_ned(turn_east);
    //letting yaw settle
    sleep_for(std::chrono::seconds(3));

    //setting controller gains per axis (general tuning) for high accuracy and low overshoot
    double kp_vx = 0.33;
    double ki_vx = 0.00005;

    double kp_vy = 0.33;
    double ki_vy = 0.00005;

    double kp_vz = 0.33;
    double ki_vz = 0.00005;

    //setting controller tolerance and intitalizing valules
    double tolerance = 0.05;
    double integral_error_x = 0;
    double integral_error_y = 0;
    double integral_error_z = 0;
    //starting timeclock
    auto previous_time = std::chrono::steady_clock::now();
    
    //loop until error is within a tolerance, send velocity control to the drone to fly
    while(euclidean_distance > tolerance){
      //getting timestep
      auto current_time = std::chrono::steady_clock::now();
      std::chrono::duration<double> elapsed_time = current_time - previous_time;
      double dt = elapsed_time.count();
      //get control values from PI controller
      std::vector<double> ux = velocity_pi_controller(kp_vx, ki_vx, dx, integral_error_x, dt);
      std::vector<double> uy = velocity_pi_controller(kp_vy, ki_vy, dy, integral_error_y, dt);
      std::vector<double> uz = velocity_pi_controller(kp_vz, ki_vz, dz, integral_error_z, dt);
      
      //updating parameters from PI controller output
      double speed = 1; // in m/s
      double vx = ux[0];
      double vy = uy[0];
      double vz = uz[0];
      integral_error_x = ux[1];
      integral_error_y = uy[1];
      integral_error_z = uz[1]; 
      
      //setting offboard commands with values
      mavsdk::Offboard::VelocityNedYaw velocity_ned_yaw{};
      velocity_ned_yaw.north_m_s = vx;
      velocity_ned_yaw.east_m_s = vy;
      velocity_ned_yaw.down_m_s = vz;
      //reuse target yaw to move in straight line
      velocity_ned_yaw.yaw_deg = target_yaw;
      offboard.set_velocity_ned(velocity_ned_yaw);
      
      //updating time
      previous_time = current_time;
      //sleep for short amount of time before calculating the new displacements and error in the loop
      sleep_for(std::chrono::milliseconds(100));
      //updating position

      position = telemetry.position_velocity_ned();
      dx = target_x_pos - position.position.north_m;
      dy = target_y_pos - position.position.east_m;
      dz = -target_z_pos - position.position.down_m;
      //terminal state is when the new euclidean distance is within tolerance 
      euclidean_distance = std::sqrt(pow(dx,2)+pow(dy,2)+pow(dz,2));

    }

    //let settle for a second before continuing to stabilize the system
    sleep_for(std::chrono::seconds(2));
  }

    //ensures that the full sequence of offboard commands have been flown.
    if (offboard_result != mavsdk::Offboard::Result::Success) {
        std::cerr << "Offboard stop failed: " << offboard_result << '\n';
        return 1;
    }
    std::cout << "Offboard stopped\n";
    return 0;
}

//function that flies a mission over a set of waypoints using latitude, longitude, and altitude
int fly_mission(mavsdk::Mission& mission, mavsdk::Mission::MissionPlan& mission_plan){
  //allows for pausing and resuming the mission mid flight
  std::atomic<bool> want_to_pause{false};

  //subscribe to mission progress before start and update status
  mission.subscribe_mission_progress([&want_to_pause](mavsdk::Mission::MissionProgress mission_progress) {
  std::cout << "Mission status update: " << mission_progress.current << " / "
            << mission_progress.total << '\n';
  });

  //starts the mission
  mavsdk::Mission::Result start_mission_result = mission.start_mission();

  //checks to see if the mission was successful, returns 1 otherwise
  if (start_mission_result != mavsdk::Mission::Result::Success) {
    std::cerr << "Starting mission failed: " << start_mission_result << '\n';
    return 1;
  }

  //adds delay to ensure mission completes before exiting
  while (!mission.is_mission_finished().second) {
    sleep_for(std::chrono::seconds(1));
  }
  
  return 0;
}