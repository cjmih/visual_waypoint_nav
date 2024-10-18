#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>

#include "detect_and_track.hpp"

cv::Mat accumulate_rotation(const cv::Mat &prev_cumulative_R, const cv::Mat &current_R)
{
  return current_R * prev_cumulative_R;
}

void decomposeHomographyMat(const cv::Mat &H, const cv::Mat &K, std::vector<cv::Mat> &rotations, std::vector<cv::Mat> &translations, std::vector<cv::Mat> &normals)
{
  cv::decomposeHomographyMat(H, K, rotations, translations, normals);

  std::cout << "Possible solutions:" << '\n';
  for (size_t i = 0; i < rotations.size(); i++)
  {
    std::cout << "Solution " << i + 1 << ":" << '\n';
    std::cout << "Rotation:" << '\n'
              << rotations[i] << '\n';
    std::cout << "Translation:" << '\n'
              << translations[i] << '\n';
    std::cout << "Normal:" << '\n'
              << normals[i] << '\n';
  }
}
