#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

double average(std::vector<float> a);

cv::Mat preprocess(const cv::Mat &input);

#endif