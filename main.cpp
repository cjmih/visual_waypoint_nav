#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>

#include "detect_and_track.hpp"
#include "prefilter.hpp"
#include "utils.hpp"

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    cout << "Usage: " << argv[0] << " <video_file>" << endl;
    return -1;
  }

  VideoCapture cap(argv[1]);
  if (!cap.isOpened())
  {
    cout << "Error opening video file" << endl;
    return -1;
  }

  Mat frame, prev_frame, gray, prev_gray;

  // for heading animation window
  cv::Mat heading(400, 400, CV_8UC3, cv::Scalar(255, 255, 255));
  vector<Point2f> prev_points, points;
  vector<KeyPoint> keypoints, prev_keypoints;
  vector<uchar> status;
  vector<float> err;

  // Parameters for goodFeaturesToTrack
  int maxCorners = 500;
  double qualityLevel = 0.01;
  double minDistance = 10;

  int frameCount = 0;

  // Camera intrinsic matrix (you should calibrate your camera to get accurate values)
  cv::Mat K = (cv::Mat_<double>(3, 3) << 1022.843511, 0.000000, 856.477662, 0.000000, 1023.550413, 472.570913, 0.000000, 0.000000, 1.000000);
  cv::Mat cumulative_R = cv::Mat::eye(3, 3, CV_64F); // Identity matrix to start

  while (true)
  {
    cap >> frame;
    if (frame.empty())
      break;

    cvtColor(frame, gray, COLOR_BGR2GRAY);

    if (prev_gray.empty())
    {
      // First frame: detect features
      goodFeaturesToTrack(gray, points, maxCorners, qualityLevel, minDistance);
      // Convert points to keypoints
      for (const auto &pt : points)
      {
        keypoints.push_back(KeyPoint(pt, 1.0));
      }
    }
    else
    {
      // Subsequent frames: track features
      if (!prev_points.empty())
      {
        vector<Point2f> current_points;
        calcOpticalFlowPyrLK(prev_gray, gray, prev_points, current_points, status, err);

        // Filter out bad points and create matched point sets
        vector<Point2f> matched_prev_points, matched_current_points;
        for (size_t i = 0; i < status.size(); i++)
        {
          if (status[i])
          {
            matched_prev_points.push_back(prev_points[i]);
            matched_current_points.push_back(current_points[i]);
            circle(frame, current_points[i], 3, Scalar(0, 255, 0), -1);
          }
        }

        // Update points and keypoints for the next iteration
        points = matched_current_points;
        keypoints.clear();
        for (const auto &pt : points)
        {
          keypoints.push_back(KeyPoint(pt, 1.0));
        }

        // Compute homography if we have enough points
        if (matched_prev_points.size() >= 4 && matched_current_points.size() >= 4)
        {
          try
          {
            Mat H = findHomography(matched_prev_points, matched_current_points, RANSAC);
            if (!H.empty() && H.rows == 3 && H.cols == 3)
            {
              std::vector<cv::Mat> rotations, translations, normals;
              decomposeHomographyMat(H, K, rotations, translations, normals);
              if (!rotations.empty())
              {
                cv::Mat current_R = rotations[0]; // Using the first solution
                cumulative_R = accumulate_rotation(cumulative_R, current_R);
                cv::Mat cumulative_euler = rotation_matrix_to_euler_angles(cumulative_R);
                cv::Mat relative_euler = rotation_matrix_to_euler_angles(current_R);

                cv::putText(frame, "Cumulative Yaw: " + std::to_string(cumulative_euler.at<double>(0)), cv::Point(20, 70),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
                // cv::putText(frame, "Cumulative Pitch: " + std::to_string(cumulative_euler.at<double>(1)), cv::Point(20, 100),
                //             cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
                // cv::putText(frame, "Cumulative Roll: " + std::to_string(cumulative_euler.at<double>(2)), cv::Point(20, 130),
                //             cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);

                cv::putText(frame, "Rel Yaw: " + std::to_string(relative_euler.at<double>(0)), cv::Point(20, 160),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                cv::putText(frame, "Rel Pitch: " + std::to_string(relative_euler.at<double>(1)), cv::Point(20, 190),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                cv::putText(frame, "Rel Roll: " + std::to_string(relative_euler.at<double>(2)), cv::Point(20, 220),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

                drawYawAnimation(heading, cumulative_euler.at<double>(0));
              }
              else
              {
                cv::putText(frame, "No valid rotation matrix", cv::Point(20, 70),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
              }
            }
            else
            {
              cv::putText(frame, "Invalid homography matrix", cv::Point(20, 70),
                          cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
            }
          }
          catch (const cv::Exception &e)
          {
            std::cout << "OpenCV exception: " << e.what() << std::endl;
            cv::putText(frame, "Homography calculation failed", cv::Point(20, 70),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
          }
        }
        else
        {
          cv::putText(frame, "Not enough points for homography", cv::Point(20, 70),
                      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        }
      }

      // If we lost too many points or it's time to refresh, detect new features
      if (points.size() < 50 || frameCount % 30 == 0)
      {
        cout << "Detecting new features" << endl;
        goodFeaturesToTrack(gray, points, maxCorners, qualityLevel, minDistance);
        keypoints.clear();
        for (const auto &pt : points)
        {
          keypoints.push_back(KeyPoint(pt, 1.0));
        }
      }
    }

    // Detect edges (for runway detection)
    Mat edges;
    Canny(gray, edges, 50, 150);

    // Display frame number and number of tracked points
    putText(frame, "Frame: " + to_string(frameCount) + " Points: " + to_string(points.size()),
            Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);

    imshow("Runway Features", frame);
    imshow("Heading Animation", heading);
    // waitKey(500);
    if (waitKey(30) >= 0)
      break;
    prev_gray = gray.clone();
    prev_points = points;
    prev_keypoints = keypoints;
    frameCount++;
  }

  cap.release();
  destroyAllWindows();
  return 0;
}