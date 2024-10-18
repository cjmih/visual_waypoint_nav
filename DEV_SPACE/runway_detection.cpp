#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

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

cv::Mat accumulate_rotation(const cv::Mat &prev_cumulative_R, const cv::Mat &current_R)
{
  return current_R * prev_cumulative_R;
}

void decomposeHomographyMat(const Mat &H, const Mat &K, vector<Mat> &rotations, vector<Mat> &translations, vector<Mat> &normals)
{
  cv::decomposeHomographyMat(H, K, rotations, translations, normals);

  cout << "Possible solutions:" << endl;
  for (size_t i = 0; i < rotations.size(); i++)
  {
    cout << "Solution " << i + 1 << ":" << endl;
    cout << "Rotation:" << endl
         << rotations[i] << endl;
    cout << "Translation:" << endl
         << translations[i] << endl;
    cout << "Normal:" << endl
         << normals[i] << endl;
  }
}

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