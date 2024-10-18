#include <iostream>
#include <opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

using namespace std;
using namespace cv;
double average(vector<float> a){
  int sum = 0; 
  int n = a.size();

  for(int i = 0; i < n; i++){
    sum += a[i];
  }
  return (double)sum/n;
}

cv::Mat preprocess(const cv::Mat& input);

int main(int argc, char **argv) {
  if (argc != 3) {
    cout << "usage: feature_extraction img1 img2" << endl;
    return 1;
  }
  Mat img_1 = imread(argv[1]);
  Mat img_2 = imread(argv[2]);

  assert(img_1.data != nullptr && img_2.data != nullptr);
  // Resize to a common resolution
  //cv::resize(img_1, img_1, cv::Size(800, 600));
  //cv::resize(img_2, img_2, cv::Size(800, 600));
  
    //     // Combine original image with edges
//   img_1 = img_1 * 0.7 + edges0 * 0.3;
//   img_2 = img_2 * 0.7 + edges1 * 0.3;
  cv::Mat mod0 = preprocess(img_1);
  cv::Mat mod1 = preprocess(img_2);
  cv::Size sz = img_2.size();
  double height = sz.height;
  double width = sz.width;
  double cx = width/2;
  double cy = height/2;


  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches, good_matches;

  Mat descriptors_1, descriptors_2;
  cv::Ptr<cv::Feature2D> detector = cv::ORB::create(3000); // Increase the number of features
  //Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  
  detector->detect(mod0, keypoints_1);
  detector->detect(mod1, keypoints_2);

  descriptor->compute(mod0, keypoints_1, descriptors_1);
  descriptor->compute(mod1, keypoints_2, descriptors_2);

  vector<vector<DMatch>> knn_matches;
  matcher->knnMatch(descriptors_1, descriptors_2, knn_matches, 2);

  const float ratio_thresh = 0.7f;
  for (size_t i = 0; i < knn_matches.size(); i++) {
    if (knn_matches[i].size() >= 2) {
      if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
        good_matches.push_back(knn_matches[i][0]);
      }
    }
  }
  vector<float> x_pose, y_pose;

  for (size_t i = 0; i < good_matches.size(); i++) {
    const DMatch& match = good_matches[i];
    
    // Retrieve the keypoints from the keypoints vectors
    const KeyPoint& keypoint_1 = keypoints_1[match.queryIdx];
    const KeyPoint& keypoint_2 = keypoints_2[match.trainIdx];
    // add the positions of good matches into an array
    if(keypoint_2.pt.x < cx+200 && keypoint_2.pt.x > cx-200){
      if(keypoint_2.pt.y < cy+200 && keypoint_2.pt.y > cy-200){
            x_pose.push_back(keypoint_1.pt.x);
            y_pose.push_back(keypoint_1.pt.y);
      }
    }


    
  }
  double avg_x = average(x_pose);
  double avg_y = average(y_pose);

  cout << "average x coordinate: " << avg_x << endl;
  cout << "average y coordinate: " << avg_y << endl;

  cout << "Total matches before ratio test: " << knn_matches.size() << endl;
  cout << "Good matches after ratio test: " << good_matches.size() << endl;
  // Mat descriptors_1, descriptors_2;
  // Ptr<FeatureDetector> detector = ORB::create();
  // Ptr<DescriptorExtractor> descriptor = ORB::create();
  // Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

  // chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  // detector->detect(img_1, keypoints_1);
  // detector->detect(img_2, keypoints_2);

  // descriptor->compute(img_1, keypoints_1, descriptors_1);
  // descriptor->compute(img_2, keypoints_2, descriptors_2);
  // chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  // chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  // cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;

  // Mat outimg1;
  // drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
  // //imshow("ORB features", outimg1);

  // vector<DMatch> matches;
  // t1 = chrono::steady_clock::now();
  // matcher->match(descriptors_1, descriptors_2, matches);
  // t2 = chrono::steady_clock::now();
  // time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  // cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;

  // auto min_max = minmax_element(matches.begin(), matches.end(),
  //                               [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
  // double min_dist = min_max.first->distance;
  // double max_dist = min_max.second->distance;

  // printf("-- Max dist : %f \n", max_dist);
  // printf("-- Min dist : %f \n", min_dist);

  // for (int i = 0; i < descriptors_1.rows; i++) {
  //   if (matches[i].distance <= max(2 * min_dist, 30.0)) {
  //     good_matches.push_back(matches[i]);
  //   }
  // }

  Mat img_match, img_goodmatch, resized, estimate, resized_estimate; 
  int width1 = 1440;
  int height1 = 720;
  
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);

  //drawing circle for radius of interest
  Point center(avg_x,avg_y);
  int radius = 100; 
  Scalar line_color(255,255,0);
  int thickness = 5; 
  estimate = img_1;
  // drawKeypoints(img1, keypoints_1, img_out, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
  circle(estimate, center, radius, line_color, thickness);


  //imshow("all matches", img_match);
  resize(img_goodmatch, resized, Size(width1,height1), INTER_LINEAR);
  resize(estimate, resized_estimate, Size(width1, height1), INTER_LINEAR);

  imwrite("results_matching.jpg", resized);  

  //imshow("good matches", img_goodmatch);
  imshow("Good Match Results", resized);
  imshow("test", img_1);

  imshow("test2", img_2);

  imshow("Estimated Position in Pixels: ", resized_estimate);
  waitKey(0);

  return 0;
}


cv::Mat preprocess(const cv::Mat& input){
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
  for (const auto& contour : contours) {
      if (cv::contourArea(contour) > 1000) { // Adjust this threshold as needed
          cv::drawContours(mask, std::vector<std::vector<cv::Point>>{contour}, 0, cv::Scalar(255), -1);
      }
  }
  cv::imshow("maskedimage", mask);
  waitKey(0);
    // Combine original image with the mask
  cv::Mat masked;
  input.copyTo(masked, mask);
  //return masked;
  return gauss;
}