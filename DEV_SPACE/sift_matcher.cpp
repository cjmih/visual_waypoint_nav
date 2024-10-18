#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <set>
#include <algorithm>

using namespace cv;
using namespace std;

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <set>
#include <algorithm>

using namespace cv;
using namespace std;

struct TrackedFeature {
    int id;
    Point2f current_pos;
    Point2f prev_pos;
    float radius;
    Scalar color;
};

class FeatureTracker {
private:
    int nextId;
    vector<TrackedFeature> tracked_features;
    float max_distance;
    set<int> previous_frame_ids;
    set<int> current_frame_ids;

public:
    FeatureTracker(float maxDist = 50) : nextId(0), max_distance(maxDist) {}

    void update(const vector<KeyPoint>& detected_keypoints, const vector<DMatch>& matches, const vector<KeyPoint>& prev_keypoints) {
        vector<TrackedFeature> new_tracked_features;
        current_frame_ids.clear();

        for (const auto& match : matches) {
            const KeyPoint& prev_kp = prev_keypoints[match.queryIdx];
            const KeyPoint& curr_kp = detected_keypoints[match.trainIdx];
            
            auto it = find_if(tracked_features.begin(), tracked_features.end(),
                              [&prev_kp](const TrackedFeature& tf) { return tf.current_pos == prev_kp.pt; });
            
            if (it != tracked_features.end()) {
                // Update existing feature
                it->prev_pos = it->current_pos;
                it->current_pos = curr_kp.pt;
                new_tracked_features.push_back(*it);
                current_frame_ids.insert(it->id);
            } else {
                // Add new feature
                int newId = nextId++;
                new_tracked_features.push_back({
                    newId,
                    curr_kp.pt,
                    prev_kp.pt,
                    curr_kp.size / 2,
                    Scalar(rand() & 255, rand() & 255, rand() & 255)
                });
                current_frame_ids.insert(newId);
            }
        }

        tracked_features = new_tracked_features;
    }

    const vector<TrackedFeature>& getTrackedFeatures() const {
        return tracked_features;
    }

    set<int> getTrackedIds() const {
        set<int> intersection;
        set_intersection(previous_frame_ids.begin(), previous_frame_ids.end(),
                         current_frame_ids.begin(), current_frame_ids.end(),
                         inserter(intersection, intersection.begin()));
        return intersection;
    }

    vector<Point2f> getPreviousPoints() const {
        vector<Point2f> points;
        for (const auto& feature : tracked_features) {
            points.push_back(feature.prev_pos);
        }
        return points;
    }

    vector<Point2f> getCurrentPoints() const {
        vector<Point2f> points;
        for (const auto& feature : tracked_features) {
            points.push_back(feature.current_pos);
        }
        return points;
    }
};

// ... (keep the other helper functions like euler_to_rotation_matrix, drawYawAnimation, etc.)

cv::Mat euler_to_rotation_matrix(const cv::Mat& euler) {
    double x = euler.at<double>(2) * CV_PI / 180.0; // Roll
    double y = euler.at<double>(1) * CV_PI / 180.0; // Pitch
    double z = euler.at<double>(0) * CV_PI / 180.0; // Yaw

    cv::Mat Rx = (cv::Mat_<double>(3,3) << 
        1, 0, 0,
        0, cos(x), -sin(x),
        0, sin(x), cos(x));

    cv::Mat Ry = (cv::Mat_<double>(3,3) << 
        cos(y), 0, sin(y),
        0, 1, 0,
        -sin(y), 0, cos(y));

    cv::Mat Rz = (cv::Mat_<double>(3,3) << 
        cos(z), -sin(z), 0,
        sin(z), cos(z), 0,
        0, 0, 1);

    return Rz * Ry * Rx;
}

void drawYawAnimation(cv::Mat& image, float yaw) {
    int centerX = image.cols / 2;
    int centerY = image.rows / 2;
    int radius = std::min(centerX, centerY) - 10;
    
    image = cv::Scalar(255, 255, 255);
    cv::circle(image, cv::Point(centerX, centerY), radius, cv::Scalar(0, 0, 0), 2);
    
    float radians = yaw * CV_PI / 180.0;
    int endX = centerX + static_cast<int>(radius * sin(radians));
    int endY = centerY - static_cast<int>(radius * cos(radians));
    cv::line(image, cv::Point(centerX, centerY), cv::Point(endX, endY), cv::Scalar(0, 0, 255), 2);
    
    cv::putText(image, "N", cv::Point(centerX - 5, centerY - radius - 10), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    cv::putText(image, "E", cv::Point(centerX + radius + 10, centerY), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    cv::putText(image, "S", cv::Point(centerX - 5, centerY + radius + 20), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    cv::putText(image, "W", cv::Point(centerX - radius - 20, centerY), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
    cv::putText(image, "Yaw: " + std::to_string(yaw), cv::Point(10, 20), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
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
    euler.at<double>(0) = z * 180.0 / CV_PI; // yaw
    euler.at<double>(1) = y * 180.0 / CV_PI; // pitch
    euler.at<double>(2) = x * 180.0 / CV_PI; // roll

    return euler;
}

cv::Mat accumulate_rotation(const cv::Mat& prev_cumulative_R, const cv::Mat& current_R) {
    return current_R * prev_cumulative_R;
}

void preprocessImage(Mat& img) {
    if (img.channels() == 3) {
        cvtColor(img, img, COLOR_BGR2GRAY);
    }
    GaussianBlur(img, img, Size(3, 3), 0);
    
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    clahe->apply(img, img);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <video_file>" << endl;
        return -1;
    }

    VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        cout << "Error opening video file" << endl;
        return -1;
    }

    Mat frame, prev_frame, gray, prev_gray;
    cv::Mat heading(400, 400, CV_8UC3, cv::Scalar(255, 255, 255));

    FeatureTracker tracker;
    Ptr<SIFT> detector = SIFT::create(2000);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

    int frameCount = 0;
    vector<KeyPoint> prev_keypoints;
    Mat prev_descriptors;

    // Camera intrinsic matrix (you should calibrate your camera to get accurate values)
    cv::Mat K = (cv::Mat_<double>(3,3) << 1022.843511, 0.000000, 856.477662, 0.000000, 1023.550413, 472.570913, 0.000000, 0.000000, 1.000000);
    cv::Mat cumulative_R = cv::Mat::eye(3, 3, CV_64F);  // Identity matrix to start
    
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        vector<KeyPoint> keypoints;
        Mat descriptors;
        detector->detectAndCompute(gray, noArray(), keypoints, descriptors);

        if (!prev_gray.empty()) {
            vector<DMatch> matches;
            matcher->match(prev_descriptors, descriptors, matches);

            // Filter matches based on distance
            double max_dist = 0; double min_dist = 100;
            for (int i = 0; i < prev_descriptors.rows; i++) {
                double dist = matches[i].distance;
                if (dist < min_dist) min_dist = dist;
                if (dist > max_dist) max_dist = dist;
            }
            vector<DMatch> good_matches;
            for (int i = 0; i < prev_descriptors.rows; i++) {
                if (matches[i].distance <= max(2*min_dist, 0.02)) {
                    good_matches.push_back(matches[i]);
                }
            }

            tracker.update(keypoints, good_matches, prev_keypoints);

            vector<Point2f> prev_points = tracker.getPreviousPoints();
            vector<Point2f> curr_points = tracker.getCurrentPoints();

            if (prev_points.size() >= 4 && curr_points.size() >= 4) {
                try {
                    Mat H = findHomography(prev_points, curr_points, RANSAC);
                    if (!H.empty() && H.rows == 3 && H.cols == 3) {
                        std::vector<cv::Mat> rotations, translations, normals;
                        cv::decomposeHomographyMat(H, K, rotations, translations, normals);
                        if (!rotations.empty()) {
                            cv::Mat current_R = rotations[0];  // Using the first solution
                            cumulative_R = accumulate_rotation(cumulative_R, current_R);
                            cv::Mat cumulative_euler = rotation_matrix_to_euler_angles(cumulative_R);
                            cv::Mat relative_euler = rotation_matrix_to_euler_angles(current_R);

                            cv::putText(frame, "Cumulative Yaw: " + std::to_string(cumulative_euler.at<double>(0)), cv::Point(20, 70),
                                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
                            cv::putText(frame, "Rel Yaw: " + std::to_string(relative_euler.at<double>(0)), cv::Point(20, 160),
                                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                            cv::putText(frame, "Rel Pitch: " + std::to_string(relative_euler.at<double>(1)), cv::Point(20, 190),
                                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                            cv::putText(frame, "Rel Roll: " + std::to_string(relative_euler.at<double>(2)), cv::Point(20, 220),
                                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

                            drawYawAnimation(heading, cumulative_euler.at<double>(0));
                        }
                    }
                } catch (const cv::Exception& e) {
                    std::cout << "OpenCV exception: " << e.what() << std::endl;
                }
            }
        }

        for (const auto& feature : tracker.getTrackedFeatures()) {
            circle(frame, feature.current_pos, feature.radius, feature.color, -1);
            putText(frame, to_string(feature.id), feature.current_pos, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
        }

        putText(frame, "Frame: " + to_string(frameCount) + " Features: " + to_string(tracker.getTrackedFeatures().size()),
                Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);

        imshow("Feature Tracking", frame);
        imshow("Heading Animation", heading);

        if (waitKey(30) >= 0) break;

        prev_gray = gray.clone();
        prev_keypoints = keypoints;
        prev_descriptors = descriptors.clone();
        frameCount++;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
