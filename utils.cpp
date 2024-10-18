#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>

#include "utils.hpp"

cv::Mat euler_to_rotation_matrix(const cv::Mat &euler)
{
  double x = euler.at<double>(2) * CV_PI / 180.0; // Roll
  double y = euler.at<double>(1) * CV_PI / 180.0; // Pitch
  double z = euler.at<double>(0) * CV_PI / 180.0; // Yaw

  cv::Mat Rx = (cv::Mat_<double>(3, 3) << 1, 0, 0,
                0, cos(x), -sin(x),
                0, sin(x), cos(x));

  cv::Mat Ry = (cv::Mat_<double>(3, 3) << cos(y), 0, sin(y),
                0, 1, 0,
                -sin(y), 0, cos(y));

  cv::Mat Rz = (cv::Mat_<double>(3, 3) << cos(z), -sin(z), 0,
                sin(z), cos(z), 0,
                0, 0, 1);

  return Rz * Ry * Rx;
}

void drawYawAnimation(cv::Mat &image, float yaw)
{
  int centerX = image.cols / 2;
  int centerY = image.rows / 2;
  int radius = std::min(centerX, centerY) - 10;

  // Clear the image
  image = cv::Scalar(255, 255, 255);

  // Draw a circle
  cv::circle(image, cv::Point(centerX, centerY), radius, cv::Scalar(0, 0, 0), 2);

  // Draw a line indicating the yaw
  float radians = yaw * CV_PI / 180.0;
  int endX = centerX + static_cast<int>(radius * sin(radians));
  int endY = centerY - static_cast<int>(radius * cos(radians));
  cv::line(image, cv::Point(centerX, centerY), cv::Point(endX, endY), cv::Scalar(0, 0, 255), 2);

  // Add text labels
  cv::putText(image, "N", cv::Point(centerX - 5, centerY - radius - 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
  cv::putText(image, "E", cv::Point(centerX + radius + 10, centerY),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
  cv::putText(image, "S", cv::Point(centerX - 5, centerY + radius + 20),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
  cv::putText(image, "W", cv::Point(centerX - radius - 20, centerY),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

  // Add yaw value text
  cv::putText(image, "Yaw: " + std::to_string(yaw), cv::Point(10, 20),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
}

cv::Mat rotation_matrix_to_euler_angles(const cv::Mat &R)
{
  double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));
  bool singular = sy < 1e-6;
  double x, y, z;
  if (!singular)
  {
    x = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
    y = std::atan2(-R.at<double>(2, 0), sy);
    z = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
  }
  else
  {
    x = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
    y = std::atan2(-R.at<double>(2, 0), sy);
    z = 0;
  }
  cv::Mat euler(3, 1, CV_64F);
  euler.at<double>(0) = z * 180.0 / CV_PI; // yaw
  euler.at<double>(1) = y * 180.0 / CV_PI; // pitch
  euler.at<double>(2) = x * 180.0 / CV_PI; // roll

  return euler;
}
