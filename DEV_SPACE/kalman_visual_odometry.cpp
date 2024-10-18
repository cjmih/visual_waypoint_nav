#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
//#include <opencv2/core/eigen.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <Eigen/Dense>
#include <iostream>
#include<opencv2/opencv.hpp>
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
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <Eigen/Dense>

class ExtendedKalmanFilter {
private:
    Eigen::VectorXd x; // State vector
    Eigen::MatrixXd P; // State covariance matrix
    Eigen::MatrixXd Q; // Process noise covariance
    Eigen::MatrixXd R; // Measurement noise covariance

public:
    ExtendedKalmanFilter(int state_dim, int measurement_dim) {
        x = Eigen::VectorXd::Zero(state_dim);
        P = Eigen::MatrixXd::Identity(state_dim, state_dim);
        Q = Eigen::MatrixXd::Identity(state_dim, state_dim) * 0.00001;
        R = Eigen::MatrixXd::Identity(measurement_dim, measurement_dim) * 0.01;
    }

    void predict(const Eigen::MatrixXd& F) {
        x = F * x;
        P = F * P * F.transpose() + Q;
    }

    void update(const Eigen::VectorXd& z, const Eigen::MatrixXd& H) {
        Eigen::VectorXd y = z - H * x;
        Eigen::MatrixXd S = H * P * H.transpose() + R;
        Eigen::MatrixXd K = P * H.transpose() * S.inverse();
        
        x = x + K * y;
        P = (Eigen::MatrixXd::Identity(x.size(), x.size()) - K * H) * P;
    }

    Eigen::VectorXd getState() const {
        return x;
    }
};

void find_feature_matches(const cv::Mat &img_1, 
                          const cv::Mat &img_2,
                          std::vector<cv::KeyPoint> &keypoints_1,
                          std::vector<cv::KeyPoint> &keypoints_2,
                          std::vector<cv::DMatch> &matches);

std::pair<cv::Mat, cv::Mat> pose_2d2d(const std::vector<cv::KeyPoint> &keypoints_1,
                                      const std::vector<cv::KeyPoint> &keypoints_2,
                                      const std::vector<cv::DMatch> &matches);

cv::Mat rotation_matrix_to_euler_angles(const cv::Mat &R);

cv::Mat track_relative_rotation(const cv::Mat &initial_rotation, const cv::Mat &current_rotation, double &FPS);

Eigen::Vector3d rotationMatrixToEulerAngles(const cv::Mat& R);

cv::Mat eulerAnglesToRotationMatrix(const Eigen::Vector3d& euler);

std::pair<cv::Mat, cv::Mat> applyEKF(const cv::Mat& R, const cv::Mat& t, ExtendedKalmanFilter& ekf);

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <video_file>\n";
        return -1;
    }

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cout << "Cannot open video file.\n";
        return -1;
    }

    cv::Mat frame_prev, frame, resize_frame;
    if (!cap.read(frame)) {
        std::cout << "Cannot read video\n";
        return -1;
    }
    
    cv::Size sz = frame.size();
    std::cout << "Frame size: " << sz << '\n';
    int resize_width = 1920;
    int resize_height = 1080;

    cv::namedWindow("output", cv::WINDOW_NORMAL);
    cv::resizeWindow("output", resize_width, resize_height);
    
    cv::Mat R_total = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t_total = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat initial_rotation = cv::Mat::eye(3, 3, CV_64F);

    ExtendedKalmanFilter ekf(6, 6); // 6 state variables, 6 measurement variables

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
            std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
            std::vector<cv::DMatch> matches;

            find_feature_matches(frame_prev, frame, keypoints_1, keypoints_2, matches);
            
            cv::Mat R, t;
            std::tie(R, t) = pose_2d2d(keypoints_1, keypoints_2, matches);

            // Apply EKF
            cv::Mat filtered_R, filtered_t;
            std::tie(filtered_R, filtered_t) = applyEKF(R, t, ekf);

            // Accumulate the rotation and translation
            R_total = R * R_total;
            t_total = t_total + R_total * t;
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            double fps = 1000.0 / duration.count();

            cv::Mat relative_euler_angles = track_relative_rotation(initial_rotation, R_total, fps);
            cv::Mat filtered_euler_angles = rotation_matrix_to_euler_angles(filtered_R);
            cv::Mat relative_ekf_angles = track_relative_rotation(initial_rotation, filtered_R, fps);

            cv::resize(frame, resize_frame, cv::Size(resize_width, resize_height), 0, 0, cv::INTER_LINEAR);
            
           

            cv::putText(resize_frame, "FPS: " + std::to_string(int(fps)), cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

            // Display raw measurements
            cv::putText(resize_frame, "Raw Yaw: " + std::to_string(relative_euler_angles.at<double>(0)), cv::Point(20, 70),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
            cv::putText(resize_frame, "Raw Pitch: " + std::to_string(relative_euler_angles.at<double>(1)), cv::Point(20, 100),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
            cv::putText(resize_frame, "Raw Roll: " + std::to_string(relative_euler_angles.at<double>(2)), cv::Point(20, 130),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);

            // // Display EKF estimates
            cv::putText(resize_frame, "EKF Yaw: " + std::to_string(filtered_euler_angles.at<double>(0)), cv::Point(20, 170),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            cv::putText(resize_frame, "EKF Pitch: " + std::to_string(filtered_euler_angles.at<double>(1)), cv::Point(20, 200),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            cv::putText(resize_frame, "EKF Roll: " + std::to_string(filtered_euler_angles.at<double>(2)), cv::Point(20, 230),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

            // Display raw translation
            cv::putText(resize_frame, "Raw X: " + std::to_string(t_total.at<double>(0)), cv::Point(20, 270),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
            cv::putText(resize_frame, "Raw Y: " + std::to_string(t_total.at<double>(1)), cv::Point(20, 300),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
            cv::putText(resize_frame, "Raw Z: " + std::to_string(t_total.at<double>(2)), cv::Point(20, 330),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

            // Display EKF translation
            // cv::putText(resize_frame, "EKF X: " + std::to_string(filtered_t.at<double>(0)), cv::Point(20, 370),
            //             cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 255), 2);
            // cv::putText(resize_frame, "EKF Y: " + std::to_string(filtered_t.at<double>(1)), cv::Point(20, 400),
            //             cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 255), 2);
            // cv::putText(resize_frame, "EKF Z: " + std::to_string(filtered_t.at<double>(2)), cv::Point(20, 430),
            //             cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 255), 2);
            cv::drawKeypoints(resize_frame, keypoints_1, resize_frame, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

            cv::imshow("output", resize_frame);
        }
        catch (const cv::Exception& e) {
            std::cerr << "OpenCV error in main loop: " << e.what() << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Standard exception in main loop: " << e.what() << std::endl;
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
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    //cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    //cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
    
    detector->detectAndCompute(img_1, cv::noArray(), keypoints_1, descriptors_1);
    detector->detectAndCompute(img_2, cv::noArray(), keypoints_2, descriptors_2);

    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descriptors_1, descriptors_2, knn_matches, 2);

    const float ratio_thresh = 0.75f;
    matches.clear();
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            matches.push_back(knn_matches[i][0]);
        }
    }
        std::vector<cv::DMatch> good_matches;
    for (const auto& match : matches) {
        if (match.distance < 0.7 * matches[matches.size()/2].distance) {
            good_matches.push_back(match);
        }
    }
    matches = good_matches;
}

std::pair<cv::Mat, cv::Mat> pose_2d2d(const std::vector<cv::KeyPoint> &keypoints_1, 
                                      const std::vector<cv::KeyPoint> &keypoints_2,
                                      const std::vector<cv::DMatch> &matches) {
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
        
        if (!E.isContinuous()) {
            E = E.clone();
        }
        
        int inliers = cv::recoverPose(E, points1, points2, K, R, t, mask);
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        return std::make_pair(cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F));
    }

    if (!R.isContinuous()) {
        R = R.clone();
    }
    if (!t.isContinuous()) {
        t = t.clone();
    }

    return std::make_pair(R, t);
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

cv::Mat track_relative_rotation(const cv::Mat &initial_rotation, const cv::Mat &current_rotation, double &FPS) {
    cv::Mat relative_rotation = current_rotation + initial_rotation.t()*(1/FPS);
    return rotation_matrix_to_euler_angles(relative_rotation);
}

cv::Mat eulerAnglesToRotationMatrix(const Eigen::Vector3d& euler) {
    // Extract the angles
    double roll = euler(0);
    double pitch = euler(1);
    double yaw = euler(2);

    // Compute rotation matrices for each axis
    Eigen::Matrix3d Rx, Ry, Rz;
    Rx << 1, 0, 0,
          0, cos(roll), -sin(roll),
          0, sin(roll), cos(roll);

    Ry << cos(pitch), 0, sin(pitch),
          0, 1, 0,
          -sin(pitch), 0, cos(pitch);

    Rz << cos(yaw), -sin(yaw), 0,
          sin(yaw), cos(yaw), 0,
          0, 0, 1;

    // Combine rotations
    Eigen::Matrix3d R = Rz * Ry * Rx;

    // Convert to cv::Mat
    cv::Mat cvR(3, 3, CV_64F);
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            cvR.at<double>(i,j) = R(i,j);

    return cvR;
}

std::pair<cv::Mat, cv::Mat> applyEKF(const cv::Mat& R, const cv::Mat& t, ExtendedKalmanFilter& ekf) {
    // Manually convert cv::Mat to Eigen::Matrix3d
    Eigen::Matrix3d eigenR;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            eigenR(i, j) = R.at<double>(i, j);
        }
    }

    // Convert rotation matrix to Euler angles
    Eigen::Vector3d euler = eigenR.eulerAngles(0, 1, 2);

    // Create measurement vector [x, y, z, roll, pitch, yaw]
    Eigen::VectorXd z(6);
    z << t.at<double>(0), t.at<double>(1), t.at<double>(2), euler(0), euler(1), euler(2);

    // Simple constant velocity model
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(6, 6);
    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(6, 6);

    // Predict
    ekf.predict(F);

    // Update
    ekf.update(z, H);

    // Get the updated state
    Eigen::VectorXd state = ekf.getState();

    // Convert back to cv::Mat
    cv::Mat filtered_t = (cv::Mat_<double>(3,1) << state(0), state(1), state(2));
    Eigen::Vector3d filtered_euler(state(3), state(4), state(5));
    cv::Mat filtered_R = eulerAnglesToRotationMatrix(filtered_euler);
    //cv::addWeighted(filtered_R, 1- 0.3, R_filtered, 0.3, 0, R_filtered);
    return std::make_pair(filtered_R, filtered_t);
}