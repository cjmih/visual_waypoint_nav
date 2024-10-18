#ifndef FILE_PARSER_H
#define FILE_PARSER_H

#include <bits/stdc++.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <tuple>
#include <string>

std::map<int, std::tuple<double, double, double>> parseFile(const std::string& filename);

#endif