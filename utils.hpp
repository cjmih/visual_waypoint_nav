#ifndef UTILS_H
#define ULTILS_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>

cv::Mat euler_to_rotation_matrix(const cv::Mat& euler);

void drawYawAnimation(cv::Mat& image, float yaw);

cv::Mat rotation_matrix_to_euler_angles(const cv::Mat &R);

#endif