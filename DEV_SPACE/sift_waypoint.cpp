#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using std::cout;
using std::endl;

const char *keys =
    "{ help h |                  | Print help message. }"
    "{ input1 | box.png          | Path to input image 1. }"
    "{ input2 | box_in_scene.png | Path to input image 2. }";

int lowThreshold = 0;
const int max_lowThreshold = 100;
const int ratio = 3;
const int kernel_size = 3;

void preprocessImage(cv::Mat &img)
{
  // Apply Gaussian blur to reduce noise
  if (img.channels() > 1)
  {
    cvtColor(img, img, COLOR_BGR2GRAY);
  }
  cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0,
                    -1, 5, -1,
                    0, -1, 0);

  // cv::filter2D(img, img, -1, kernel);
  // GaussianBlur(img, img, Size(3, 3), 0);
  //  Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
  Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
  clahe->apply(img, img);

  // uncomment for canny edge detection
  // Canny(img, img, lowThreshold,lowThreshold*ratio, kernel_size);
}

double average(const std::vector<float> &a)
{
  double sum = 0;
  for (float val : a)
  {
    sum += val;
  }
  return sum / a.size();
}

void draw_circle(cv::Mat &img, const cv::Point &center, int radius)
{
  cv::Scalar line_color(255, 0, 0);
  int thickness = 5;
  cv::circle(img, center, radius, line_color, thickness);
}

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    cout << "Usage: " << argv[0] << " <path_to_image1> <path_to_image2>" << endl;
    return -1;
  }

  Mat img1 = imread(argv[1]);
  Mat img2 = imread(argv[2]);

  if (img1.empty() || img2.empty())
  {
    cout << "Could not open or find the image!\n"
         << endl;
    return -1;
  }
  cv::resize(img1, img1, cv::Size(1440, 720), 0, 0, cv::INTER_LINEAR);
  cv::resize(img2, img2, cv::Size(1440, 720), 0, 0, cv::INTER_LINEAR);
  imshow("1", img1);
  imshow("2", img2);

  // Rest of your code remains the same
  // Preprocess images
  // preprocessImage(img1);
  // preprocessImage(img2);

  // Step 1: Detect the keypoints using SIFT Detector, compute the descriptors
  int nfeatures = 0; // 0 means no limit
  int nOctaveLayers = 6;
  double contrastThreshold = 0.06;
  double edgeThreshold = 10;
  double sigma = 1.6;
  Ptr<SIFT> sift_detector = SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);

  Ptr<FeatureDetector> detector = SIFT::create();
  Ptr<DescriptorExtractor> descriptor = SIFT::create();

  std::vector<KeyPoint> keypoints1, keypoints2_good, keypoints2;
  Mat descriptors1, descriptors2;

  cv::Size sz1 = img2.size();
  int img_height = sz1.height;
  int img_width = sz1.width;
  cv::Point center(img_width / 2, img_height / 2);
  int radius = 300;
  int thickness = 5;
  cv::Scalar line_color(255, 0, 0);
  cv::Mat img_out = img1;
  int x1 = img_width / 2 + radius;
  int y1 = img_height / 2 + radius;

  // circle(img_out, center, radius, line_color, thickness);
  detector->detect(img2, keypoints2);
  detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);

  for (KeyPoint &kp : keypoints2)
  {
    double distance = sqrt(pow(kp.pt.x - center.x, 2) + pow(kp.pt.y - center.y, 2));
    if (distance <= radius)
    {
      keypoints2_good.push_back(kp);
    }
    // cout << "KeyPoint: " << kp.pt << endl;
  }

  descriptor->compute(img2, keypoints2_good, descriptors2);
  std::cout << "Size of keypoints1 in waypoint: " << keypoints2_good.size() << endl;

  // detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);

  // Step 2: Matching descriptor vectors with a FLANN based matcher
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
  std::vector<std::vector<DMatch>> knn_matches;
  matcher->knnMatch(descriptors2, descriptors1, knn_matches, 2);

  // Filter matches using the Lowe's ratio test
  const float ratio_thresh = 0.92f;
  std::vector<DMatch> good_matches;
  for (size_t i = 0; i < knn_matches.size(); i++)
  {
    if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
    {
      good_matches.push_back(knn_matches[i][0]);
    }
  }
  std::cout << "Size of good matches: " << good_matches.size() << std::endl;

  // Perform cross-checking
  std::vector<DMatch> better_matches;
  std::vector<std::vector<DMatch>> backward_matches;
  matcher->knnMatch(descriptors1, descriptors2, backward_matches, 2);
  for (size_t i = 0; i < good_matches.size(); i++)
  {
    int forward_idx = good_matches[i].trainIdx;
    int backward_idx = backward_matches[forward_idx][0].trainIdx;
    if (backward_idx == good_matches[i].queryIdx)
    {
      better_matches.push_back(good_matches[i]);
    }
  }

  // RANSAC for outlier rejection
  std::vector<Point2f> points1, points2;
  for (size_t i = 0; i < better_matches.size(); i++)
  {
    points1.push_back(keypoints1[better_matches[i].queryIdx].pt);
    points2.push_back(keypoints2_good[better_matches[i].trainIdx].pt);
  }
  std::vector<char> inliers;
  findHomography(points1, points2, RANSAC, 3.0, inliers);

  std::vector<DMatch> best_matches;
  for (size_t i = 0; i < inliers.size(); i++)
  {
    if (inliers[i])
    {
      best_matches.push_back(better_matches[i]);
    }
  }
  std::vector<float> x_pose, y_pose;
  for (const auto &match : best_matches)
  {
    x_pose.push_back(keypoints2_good[match.queryIdx].pt.x);
    y_pose.push_back(keypoints2_good[match.queryIdx].pt.y);
  }
  double avg_x = average(x_pose);
  double avg_y = average(y_pose);

  std::cout << "Size of keypoints1_good" << keypoints2_good.size() << '\n';

  std::cout << "average x coordinate: " << avg_x << std::endl;
  std::cout << "average y coordinate: " << avg_y << std::endl;
  std::cout << "Good matches: " << good_matches.size() << std::endl;
  Mat estimate, resized_estimate;
  estimate = img1.clone();
  draw_circle(estimate, cv::Point(avg_x, avg_y), 100);
  cv::imshow("Estimated Position in Pixels: ", estimate);

  // Draw matches
  Mat img_matches;
  std::cout << "Size of best matches: " << best_matches.size() << endl;

  drawMatches(img2, keypoints2_good, img1, keypoints1, best_matches, img_matches, Scalar::all(-1),
              Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  // Show detected matches
  cv::resize(img_matches, img_matches, cv::Size(1440, 720), 0, 0, cv::INTER_LINEAR);

  imshow("Best Matches", img_matches);
  waitKey();
  return 0;
}
