#ifndef DETECT_AND_TRACK_H
#define DETECT_AND_TRACK_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>

cv::Mat accumulate_rotation(const cv::Mat& prev_cumulative_R, const cv::Mat& current_R);

void decomposeHomographyMat(const cv::Mat& H, const cv::Mat& K, std::vector<cv::Mat>& rotations, std::vector<cv::Mat>& translations, std::vector<cv::Mat>& normals);

#endif