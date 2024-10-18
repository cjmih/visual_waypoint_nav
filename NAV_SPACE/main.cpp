//MAVSDK requirements
#include <mavsdk/mavsdk.h>
#include <mavsdk/plugins/action/action.h>
#include <mavsdk/plugins/mission/mission.h>
#include <mavsdk/plugins/telemetry/telemetry.h>
#include <mavsdk/plugins/offboard/offboard.h>

//general
#include <functional>
#include <future>
#include <thread>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <math.h>
#include <vector>
#include <sstream>
#include <iterator> 
#include <bits/stdc++.h>

#include "file_parser.hpp"
#include "drone_controller.hpp"

//using std::chrono::seconds;
using std::this_thread::sleep_for;

int main(int argc, char **argv){
  //check to make sure argument types are valid, otherwise throws error
  if (argc != 4) {
    usage(argv[0]);
    return 1;
  }
  //recieving the file path of the coordinates from the command line
  std::string inputfile(argv[2]);
  //recieving the control type (mission or offboard) from the user
  std::string input_type(argv[3]);
  
  //takes the input type and ensures the whole string is lowercase before checking the input type
  transform(input_type.begin(), input_type.end(), input_type.begin(), ::tolower);
  std::cout << "Input type: " << input_type << '\n';
  
  //begins connection sequence to mavsdk
  mavsdk::Mavsdk mavsdk;
  mavsdk::ConnectionResult connection_result = mavsdk.add_any_connection(argv[1]);

  //checks connection
  if (connection_result != mavsdk::ConnectionResult::Success) {
    std::cerr << "Connection failed: " << connection_result << '\n';
    return 1;
  }

  //discovers autopilot
  auto system = get_system(mavsdk);
  if (!system) {
    return 1;
  }

  //setting mavsdk modules to the system
  auto action = mavsdk::Action{system};
  auto mission = mavsdk::Mission{system};
  auto offboard = mavsdk::Offboard{system};
  auto telemetry = mavsdk::Telemetry{system};
  
  //checks telemetry connection
  while (!telemetry.health_all_ok()) {
    std::cout << "Waiting for system to be ready\n";
    sleep_for(std::chrono::seconds(1));
  }
  //print confirmation
  std::cout << "System ready\n";

  //call to file parser to take the input file and return a map of the waypoints to fly to 
  std::map<int, std::tuple<double, double, double>> flight_waypoints = parseFile(inputfile);
  //Print out the size of waypoints to ensure proper upload
  std::cout << "Size of waypoints: " << flight_waypoints.size() << '\n';
  
  //check to see if mission or offboard for flight control type
  if(input_type=="mission"){

    std::cout << "Creating and uploading mission\n";
    //parses file for flight coordinates and adds them to mission_items to be uploaded
    std::vector<mavsdk::Mission::MissionItem>mission_items = setup_mission(flight_waypoints);
    //upload new mission items to be flown
    std::cout << "Uploading mission...\n";

    //initializing and uploading mission
    mavsdk::Mission::MissionPlan mission_plan{};
    mission_plan.mission_items = mission_items;
    const mavsdk::Mission::Result upload_result = mission.upload_mission(mission_plan);
    
    //checks to make sure the mission is uploaded
    if (upload_result != mavsdk::Mission::Result::Success) {
      std::cerr << "Mission upload failed: " << upload_result << ", exiting.\n";
      return 1;
    }
    //arm drone before mission
    arm_drone(action);

    //function flies the mission and returns 0 if successful or 1 if fails.
    int mission_result = fly_mission(mission, mission_plan);
    //checks to see if mission was flown sucessfully
    if(mission_result==0){
      std::cout << "Mission flight successful!" << '\n';
    }else{
      std::cout << "Mission flight failed!!!" << '\n';
    }
    
  }else if(input_type=="offboard"){
    //getting origin position before takeoff
    mavsdk::Telemetry::PositionVelocityNed origin = telemetry.position_velocity_ned();
    std::cout << "Origin position: " << origin << '\n';
    //arming the drone
    arm_drone(action);
    
    //commanding the drone to takeoff to fixed altitude
    const auto takeoff_result = action.takeoff();
    if (takeoff_result != mavsdk::Action::Result::Success) {
        std::cerr << "Takeoff failed: " << takeoff_result << '\n';
        return 1;
    }
    
    auto in_air_promise = std::promise<void>{};
    auto in_air_future = in_air_promise.get_future();
    //confirming drone tookoff successfully
    telemetry.subscribe_landed_state([&telemetry, &in_air_promise](mavsdk::Telemetry::LandedState state) {
        if (state == mavsdk::Telemetry::LandedState::InAir) {
            std::cout << "Taking off has finished\n.";
            telemetry.subscribe_landed_state(nullptr);
            in_air_promise.set_value();
        }
    });

    //returns one if drone does not take off successfully
    in_air_future.wait_for(std::chrono::seconds(10));
    if (in_air_future.wait_for(std::chrono::seconds(3)) == std::future_status::timeout) {
        std::cerr << "Takeoff timed out.\n";
        return 1;
    }
    
    //flys positions provided from input text file, returns 1 if fails, 0 if successful
    int offboard_result = fly_offboard(offboard, telemetry, flight_waypoints, origin);
    
    if(offboard_result==0){
      std::cout << "Offboard flight successful!" << '\n';
    }else{
      std::cout << "Offboard flight failed!!!" << '\n';
    }
  }

  // sending drone RTL command to go home.
  std::cout << "Commanding RTL...\n";
  const mavsdk::Action::Result rtl_result = action.return_to_launch();
  if (rtl_result != mavsdk::Action::Result::Success) {
    std::cout << "Failed to command RTL: " << rtl_result << '\n';
    return 1;
  }
  std::cout << "Commanded RTL.\n";

  //wait for arming step
  sleep_for(std::chrono::seconds(2));

  while (telemetry.armed()) {
    // Wait until we're done.
    sleep_for(std::chrono::seconds(1));
  }
  
  std::cout << "Disarmed, exiting.\n";
}
