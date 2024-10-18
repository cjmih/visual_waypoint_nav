#include "file_parser.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <tuple>
#include <string>

//parses some inputfile and reutrn a map of waypoints with (ID -> (X, Y, Z))
std::map<int, std::tuple<double, double, double>> parseFile(const std::string& filename){
  //empty map of waypoints
  std::map<int, std::tuple<double, double, double>> waypoints;
  std::ifstream file(filename);
  //confirms that the file can be opened, throws error otherwise
  if(!file.is_open()){
    std::cerr << "Error opening the file!" << std::endl;
    return waypoints;
  }
  
  std::string line;
  //parses the file line-by-line using comma seperated delimiter
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string value;
    //declaring stored values
    int counter = 0;
    double X, Y, Z;
    int ID;
    while (std::getline(iss, value, ',')) {
      try{
        //converts inputs to respective integers and doubles
        switch(counter){
          case 0: ID = std::stoi(value); break;
          case 1: X = std::stod(value); break;
          case 2: Y = std::stod(value); break;
          case 3: Z = std::stod(value); break;
        }
        //iterates through waypoints
        counter++;

      }catch(const std::exception& e){
        std::cerr << "Error Parsing value: " << value << " " << e.what() << std::endl;
      }
      
    }
        
    //adds X,Y,Z coordinates to map with ID as the key
    //abs() ensures that the altitude is a positive value for simulator convention, we don't want our drone crashing into the Earth mission and offboard modes will use different altitude commands.
    waypoints[ID] = std::make_tuple(X,Y, abs(Z));    
  }
  
  return waypoints;
}