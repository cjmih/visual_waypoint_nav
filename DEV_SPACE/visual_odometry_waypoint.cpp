#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

void draw_circle(cv::Mat& img, const cv::Point& center, int radius);

double average(const std::vector<float>& a);

void position_estimator(cv::Mat &map, 
                        const cv::Mat &frame,
                        std::vector<cv::KeyPoint> &map_keypoints,
                        std::vector<cv::KeyPoint> &frame_keypoints,
                        std::vector<cv::DMatch> &map_matches);

void find_feature_matches(const cv::Mat &img_1, 
                          const cv::Mat &img_2,
                          std::vector<cv::KeyPoint> &keypoints_1,
                          std::vector<cv::KeyPoint> &keypoints_2,
                          std::vector<cv::DMatch> &matches);

std::pair<cv::Mat, cv::Mat> pose_2d2d(const std::vector<cv::KeyPoint> &keypoints_1,
                                      const std::vector<cv::KeyPoint> &keypoints_2,
                                      const std::vector<cv::DMatch> &matches);

void save_to_csv(const cv::Mat &R, const cv::Mat &t, const cv::Mat &euler, const std::string &filename);

cv::Mat rotation_matrix_to_euler_angles(const cv::Mat &R);

cv::Mat track_relative_rotation(const cv::Mat &initial_rotation, const cv::Mat &current_rotation);
using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;


int main(int argc, char **argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <video_file>\n";
        return -1;
    }

    cv::VideoCapture cap(argv[1]);
    cv::Mat estimate_image = cv::imread(argv[2]);
    if (!cap.isOpened()) {
        std::cout << "Cannot open video file.\n";
        return -1;
    }

    cv::Mat frame_prev, frame, resize;
    if (!cap.read(frame)) {
        std::cout << "Cannot read video\n";
        return -1;
    }
    
    cv::Size sz = frame.size();
    //std::cout << "Frame size: " << sz << '\n';
    int resize_width = 1920;
    int resize_height = 1080;

    cv::namedWindow("output", cv::WINDOW_NORMAL);
    cv::resizeWindow("output", resize_width, resize_height);
    
    cv::Mat R_total = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t_total = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat initial_rotation = cv::Mat::eye(3, 3, CV_64F);

    // Define camera mount rotation (example: camera pitched down 45 degrees)
    // cv::Mat camera_mount_rotation = (cv::Mat_<double>(3,3) <<
    //     1, 0, 0,
    //     0, cos(-45*CV_PI/180), -sin(-45*CV_PI/180),
    //     0, sin(-45*CV_PI/180), cos(-45*CV_PI/180));

    while (true) {
        auto start = std::chrono::high_resolution_clock::now();
        if (!cap.read(frame)) {
            std::cout << "End of video stream\n";
            break;
        }

        if (frame_prev.empty()) {
            frame.copyTo(frame_prev);
            continue;
        }
        
        try {
            std::vector<cv::KeyPoint> keypoints_1, keypoints_2, map_keypoints, frame_keypoints, keypoints1_good;
            std::vector<cv::DMatch> matches, map_matches;

            //find_feature_matches(frame_prev, frame, keypoints_1, keypoints_2, matches);
            cv::Mat estimate = estimate_image.clone();
            try {
                position_estimator(estimate, frame, map_keypoints, frame_keypoints, map_matches);

            } catch (const std::exception& e) {
                std::cerr << "Exception caught in position_estimator: " << e.what() << std::endl;
                continue;  // Skip to the next iteration of the main loop
            } catch (...) {
                std::cerr << "Unknown exception caught in position_estimator" << std::endl;
                continue;  // Skip to the next iteration of the main loop
            }
            if (map_matches.empty()) {
                std::cout << "No matches found in this frame. Continuing to next frame." << std::endl;
                continue;  // Skip to the next iteration of the main loop
            }
            
            //cv::Mat R, t;
            //std::tie(R, t) = pose_2d2d(keypoints_1, keypoints_2, matches);

            // Accumulate the rotation and translation
            //R_total = R * R_total;
            //t_total = t_total + R_total * t;

            //cv::Mat relative_euler_angles = track_relative_rotation(initial_rotation, R_total);
            
            // Transform to drone body frame
            //cv::Mat drone_euler_angles = camera_to_drone_frame(relative_euler_angles, camera_mount_rotation);

            // Save the data for each frame
            //save_to_csv(R_total, t_total, relative_euler_angles, "pose_data.csv");

            cv::resize(frame, resize, cv::Size(resize_width, resize_height), 0, 0, cv::INTER_LINEAR);
            
            // auto end = std::chrono::high_resolution_clock::now();
            // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            // double fps = 1000.0 / duration.count();

            // cv::putText(resize, "FPS: " + std::to_string(int(fps)), cv::Point(10, 30),
            //             cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

            // cv::putText(resize, "Rel Yaw: " + std::to_string(relative_euler_angles.at<double>(0)), cv::Point(20, 70),
            //             cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
            // cv::putText(resize, "Rel Pitch: " + std::to_string(relative_euler_angles.at<double>(1)), cv::Point(20, 100),
            //             cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
            // cv::putText(resize, "Rel Roll: " + std::to_string(relative_euler_angles.at<double>(2)), cv::Point(20, 130),
            //             cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);

            // // cv::putText(resize, "Drone Yaw: " + std::to_string(drone_euler_angles.at<double>(0)), cv::Point(20, 170),
            // //             cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
            // // cv::putText(resize, "Drone Pitch: " + std::to_string(drone_euler_angles.at<double>(1)), cv::Point(20, 200),
            // //             cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
            // // cv::putText(resize, "Drone Roll: " + std::to_string(drone_euler_angles.at<double>(2)), cv::Point(20, 230),
            // //             cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);

            // cv::putText(resize, "X: " + std::to_string(t_total.at<double>(0)), cv::Point(20, 270),
            //             cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
            // cv::putText(resize, "Y: " + std::to_string(t_total.at<double>(1)), cv::Point(20, 300),
            //             cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
            // cv::putText(resize, "Z: " + std::to_string(t_total.at<double>(2)), cv::Point(20, 330),
            //             cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

            cv::imshow("output", resize);
            cv::imshow("estimate", estimate);
        }
        catch (const cv::Exception& e) {
            std::cerr << "OpenCV error in main loop: " << e.what() << std::endl;
            // Skip this frame and continue with the next one
        }
        catch (const std::exception& e) {
            std::cerr << "Standard exception in main loop: " << e.what() << std::endl;
            // Skip this frame and continue with the next one
        }

        frame.copyTo(frame_prev);

        char c = (char)cv::waitKey(30);
        if (c == 27) // Exit if 'Esc' is pressed
            break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}

void find_feature_matches(const cv::Mat &img_1, 
                          const cv::Mat &img_2,
                          std::vector<cv::KeyPoint> &keypoints_1,
                          std::vector<cv::KeyPoint> &keypoints_2,
                          std::vector<cv::DMatch> &matches) {
    cv::Mat descriptors_1, descriptors_2;
    // cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    // cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    
    int nfeatures = 0; // 0 means no limit
    int nOctaveLayers = 3;
    double contrastThreshold = 0.04;
    double edgeThreshold = 10;
    double sigma = 1.6;
    Ptr<SIFT> detector = SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    

    // detector->detectAndCompute(img_1, cv::noArray(), keypoints_1, descriptors_1);
    // detector->detectAndCompute(img_2, cv::noArray(), keypoints_2, descriptors_2);

    // std::vector<std::vector<cv::DMatch>> knn_matches;
    // matcher->knnMatch(descriptors_1, descriptors_2, knn_matches, 2);
    detector->detectAndCompute(img_1, noArray(), keypoints_1, descriptors_1);
    detector->detectAndCompute(img_2, noArray(), keypoints_2, descriptors_2);

    // Step 2: Matching descriptor vectors with a FLANN based matcher
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch(descriptors_1, descriptors_2, knn_matches, 2);
    const float ratio_thresh = 0.75f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    // const float ratio_thresh = 0.75f;
    // matches.clear();
    // for (size_t i = 0; i < knn_matches.size(); i++) {
    //     if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
    //         matches.push_back(knn_matches[i][0]);
    //     }
    // }
    std::cout << "Size of good matches: " << matches.size() << '\n';
    
}

std::pair<cv::Mat, cv::Mat> pose_2d2d(const std::vector<cv::KeyPoint> &keypoints_1, 
                                      const std::vector<cv::KeyPoint> &keypoints_2,
                                      const std::vector<cv::DMatch> &matches) {
    //for logitec
    //cv::Mat K = (cv::Mat_<double>(3,3) << 949.385218, 0, 903.408940, 0, 946.396095, 591.981431, 0, 0, 1);
    //for dji 
    cv::Mat K = (cv::Mat_<double>(3,3) << 1022.843511, 0.000000, 856.477662, 0.000000, 1023.550413, 472.570913, 0.000000, 0.000000, 1.000000);
    
    std::vector<cv::Point2f> points1, points2;

    for (const auto& match : matches) {
        points1.push_back(keypoints_1[match.queryIdx].pt);
        points2.push_back(keypoints_2[match.trainIdx].pt);
    }
    
    if (points1.size() < 5 || points2.size() < 5) {
        std::cerr << "Not enough points to estimate Essential Matrix" << std::endl;
        return std::make_pair(cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F));
    }

    cv::Mat E, R, t, mask;

    try {
        E = cv::findEssentialMat(points1, points2, K, cv::RANSAC, 0.999, 1.0, mask);
        //std::cout << "Essential Mat: " << E << '\n';
        
        // Ensure E is continuous
        if (!E.isContinuous()) {
            E = E.clone();
        }
        
        // Decompose E to get R and t
        int inliers = cv::recoverPose(E, points1, points2, K, R, t, mask);
        //std::cout << "Number of inliers: " << inliers << '\n';
        std::cout << "R:\n" << R << "\nt:\n" << t << '\n';
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        return std::make_pair(cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F));
    }

    // Ensure R and t are continuous
    if (!R.isContinuous()) {
        R = R.clone();
    }
    if (!t.isContinuous()) {
        t = t.clone();
    }

    return std::make_pair(R, t);
}
void position_estimator(cv::Mat &map,
                        const cv::Mat &frame,
                        std::vector<cv::KeyPoint> &map_keypoints,
                        std::vector<cv::KeyPoint> &frame_keypoints,
                        std::vector<cv::DMatch> &map_matches) {

    int nfeatures = 0; // 0 means no limit
    int nOctaveLayers = 6;
    double contrastThreshold = 0.06;
    double edgeThreshold = 10;
    double sigma = 1.6;
    cv::Ptr<cv::SIFT> sift_detector = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    
    cv::Ptr<cv::Feature2D> detector = cv::SIFT::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::SIFT::create();

    cv::Mat descriptors1, descriptors2;

    detector->detect(map, map_keypoints);
    
    if (map_keypoints.empty()) {
        std::cout << "No keypoints found in the map image." << std::endl;
        return;
    }
    
    descriptor->compute(map, map_keypoints, descriptors1);
    detector->detectAndCompute(frame, cv::noArray(), frame_keypoints, descriptors2);
    
    if (frame_keypoints.empty()) {
        std::cout << "No keypoints found in the frame image." << std::endl;
        return;
    }
    
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
    
    const float ratio_thresh = 0.75f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i].size() >= 2 && knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    
    std::cout << "Number of good matches: " << good_matches.size() << std::endl;
    
    if (good_matches.empty()) {
        std::cout << "No good matches found between map and frame." << std::endl;
        return;
    }
    
    std::vector<float> x_pose, y_pose;
    for (const auto& match : good_matches) {
        if (match.queryIdx < map_keypoints.size() && match.trainIdx < frame_keypoints.size()) {
            x_pose.push_back(map_keypoints[match.queryIdx].pt.x);
            y_pose.push_back(map_keypoints[match.queryIdx].pt.y);
        }
    }

    if (!x_pose.empty() && !y_pose.empty()) {
        double avg_x = average(x_pose);
        double avg_y = average(y_pose);
        
        draw_circle(map, cv::Point(avg_x, avg_y), 100);
    } else {
        std::cout << "Unable to calculate average position." << std::endl;
    }

    // Assign good_matches to map_matches
    map_matches = good_matches;
}

void save_to_csv(const cv::Mat &R, const cv::Mat &t, const cv::Mat &euler, const std::string &filename) {
    static bool first_write = true;
    std::ofstream file;
    
    if (first_write) {
        file.open(filename);
        file << "R11,R12,R13,R21,R22,R23,R31,R32,R33,tx,ty,tz,yaw,pitch,roll" << '\n';
        first_write = false;
    } else {
        file.open(filename, std::ios::app);
    }

    if (file.is_open()) {
        // Write rotation matrix
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                file << std::setprecision(6) << R.at<double>(i, j) << ",";
            }
        }

        // Write translation vector
        file << t.at<double>(0) << "," << t.at<double>(1) << "," << t.at<double>(2) << ",";

        // Write Euler angles
        file << euler.at<double>(0) << "," << euler.at<double>(1) << "," << euler.at<double>(2) << '\n';

        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << '\n';
    }
}

cv::Mat rotation_matrix_to_euler_angles(const cv::Mat &R) {
    double sy = std::sqrt(R.at<double>(0,0) * R.at<double>(0,0) + R.at<double>(1,0) * R.at<double>(1,0));
    bool singular = sy < 1e-6;
    double x, y, z;
    if (!singular) {
        x = std::atan2(R.at<double>(2,1), R.at<double>(2,2));
        y = std::atan2(-R.at<double>(2,0), sy);
        z = std::atan2(R.at<double>(1,0), R.at<double>(0,0));
    } else {
        x = std::atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = std::atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    cv::Mat euler(3, 1, CV_64F);
    euler.at<double>(0) = y * 180.0 / CV_PI; // yaw
    euler.at<double>(1) = x * 180.0 / CV_PI; // pitch
    euler.at<double>(2) = z * 180.0 / CV_PI; // roll

    return euler;
}

cv::Mat track_relative_rotation(const cv::Mat &initial_rotation, const cv::Mat &current_rotation) {
    cv::Mat relative_rotation = current_rotation * initial_rotation.t();
    return rotation_matrix_to_euler_angles(relative_rotation);
}

double average(const std::vector<float>& a) {
    if (a.empty()) {
        return 0.0;  // or throw an exception, or return a special value
    }
    double sum = 0;
    for (float val : a) {
        sum += val;
    }
    return sum / a.size();
}

void draw_circle(cv::Mat& img, const cv::Point& center, int radius) {
    cv::Scalar line_color(255, 0, 0);
    int thickness = 5;
    cv::circle(img, center, radius, line_color, thickness);
}