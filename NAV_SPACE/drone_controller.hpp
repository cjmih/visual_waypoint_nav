#ifndef DRONE_CONTROLLER_H
#define DRONE_CONTROLLER_H

#include <vector>
#include <string>
#include <tuple>
#include <unordered_map>
#include <bits/stdc++.h>
#include <mavsdk/mavsdk.h>
#include <mavsdk/plugins/mission/mission.h>
#include <mavsdk/plugins/offboard/offboard.h>

int arm_drone(mavsdk::Action& action);

void usage(const std::string& bin_name);

std::shared_ptr<mavsdk::System> get_system(mavsdk::Mavsdk& mavsdk);

int fly_mission(mavsdk::Mission& mission, mavsdk::Mission::MissionPlan& mission_plan);

std::vector<double> velocity_pi_controller(double kp, double ki, double error, double int_error, double timestep);

std::vector<mavsdk::Mission::MissionItem> setup_mission(std::map<int, std::tuple<double, double, double>>& waypoints);

int fly_offboard(mavsdk::Offboard& offboard, mavsdk::Telemetry& telemetry, std::map<int, std::tuple<double, double, double>> flight_waypoints, mavsdk::Telemetry::PositionVelocityNed& origin);

#endif