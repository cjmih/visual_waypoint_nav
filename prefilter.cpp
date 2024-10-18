#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

#include "prefilter.hpp"


double average(std::vector<float> a)
{
  int sum = 0;
  int n = a.size();

  for (int i = 0; i < n; i++)
  {
    sum += a[i];
  }
  return (double)sum / n;
}

cv::Mat preprocess(const cv::Mat &input)
{
  cv::Mat gray, equalized, thresh, canny, dilated, eroded, gauss, cont;
  cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
  int thresh_low = 0;
  int thresh_high = 255;
  // Edge enhancement
  // cv::Mat edges;
  // cv::Laplacian(img_1, img_1, CV_8U, 3);
  // edges = edges + img_1;
  //    Apply histogram equalization
  cv::equalizeHist(gray, equalized);
  cv::threshold(equalized, thresh, thresh_low, thresh_high, cv::THRESH_BINARY | cv::THRESH_OTSU);
  // Edge detection
  cv::Canny(thresh, canny, 50, 150);
  cv::dilate(canny, dilated, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
  cv::erode(dilated, eroded, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

  cv::GaussianBlur(eroded, gauss, cv::Size(3, 3), 0.75);
  // Find contours
  std::vector<std::vector<cv::Point>> contours;

  cv::findContours(gauss, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  // Create a mask for large contours (runway)
  cv::Mat mask = cv::Mat::zeros(input.size(), CV_8UC1);
  for (const auto &contour : contours)
  {
    if (cv::contourArea(contour) > 1000)
    { // Adjust this threshold as needed
      cv::drawContours(mask, std::vector<std::vector<cv::Point>>{contour}, 0, cv::Scalar(255), -1);
    }
  }
  cv::imshow("maskedimage", mask);
  cv::waitKey(0);
  // Combine original image with the mask
  cv::Mat masked;
  input.copyTo(masked, mask);
  // return masked;
  return gauss;
}