#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <chrono>
#include <vector>
#include <map>
#include <set>
#include <fstream>

using namespace std;
using namespace cv;

struct TrackedBlob {
    int id;
    Point2f center;
    float radius;
    Scalar color;
};

class BlobTracker {
private:
    int nextId;
    vector<TrackedBlob> trackedBlobs;
    float maxDistance;
    set<int> previousFrameIds;
    set<int> currentFrameIds;

public:
    BlobTracker(float maxDist = 50.0) : nextId(0), maxDistance(maxDist) {}

    void update(const vector<KeyPoint>& detectedKeypoints) {
        vector<TrackedBlob> newTrackedBlobs;
        vector<bool> matched(detectedKeypoints.size(), false);

        previousFrameIds = currentFrameIds;
        currentFrameIds.clear();

        // Match existing tracks to new detections
        for (const auto& trackedBlob : trackedBlobs) {
            int bestMatch = -1;
            float minDist = maxDistance;

            for (size_t i = 0; i < detectedKeypoints.size(); ++i) {
                if (matched[i]) continue;
                float dist = norm(trackedBlob.center - detectedKeypoints[i].pt);
                if (dist < minDist) {
                    minDist = dist;
                    bestMatch = i;
                }
            }

            if (bestMatch != -1) {
                matched[bestMatch] = true;
                newTrackedBlobs.push_back({
                    trackedBlob.id,
                    detectedKeypoints[bestMatch].pt,
                    detectedKeypoints[bestMatch].size / 2,
                    trackedBlob.color
                });
                currentFrameIds.insert(trackedBlob.id);
            }
        }

        // Add new tracks for unmatched detections
        for (size_t i = 0; i < detectedKeypoints.size(); ++i) {
            if (!matched[i]) {
                int newId = nextId++;
                newTrackedBlobs.push_back({
                    newId,
                    detectedKeypoints[i].pt,
                    detectedKeypoints[i].size / 2,
                    Scalar(rand() & 255, rand() & 255, rand() & 255)
                });
                currentFrameIds.insert(newId);
            }
        }

        trackedBlobs = newTrackedBlobs;
    }

    const vector<TrackedBlob>& getTrackedBlobs() const {
        return trackedBlobs;
    }

    set<int> getTrackedIds() const {
        set<int> intersection;
        set_intersection(previousFrameIds.begin(), previousFrameIds.end(),
                         currentFrameIds.begin(), currentFrameIds.end(),
                         inserter(intersection, intersection.begin()));
        return intersection;
    }
};

void writeTrackedIdsToFile(const set<int>& trackedIds, ofstream& outFile) {
    outFile << "Tracked IDs: ";
    for (int id : trackedIds) {
        outFile << id << " ";
    }
    outFile << endl;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <input_video> <output_file>" << endl;
        return -1;
    }

    VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        cout << "Cannot open video file.\n";
        return -1;
    }

    ofstream outFile(argv[2]);
    if (!outFile.is_open()) {
        cout << "Cannot open output file.\n";
        return -1;
    }

    BlobTracker blobTracker;

    int frameCount = 0;
    while (1) {
        Mat frame;
        if (!cap.read(frame)) {
            cout << "Cannot read video\n";
            break;
        }
        cout << "Processing frame " << frameCount << ", size: " << frame.size() << endl;

        Mat frame_gray, blobs, masked_blobs;
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

        Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create();
        vector<KeyPoint> keypoints;
        detector->detect(frame_gray, keypoints);

        blobTracker.update(keypoints);

        set<int> trackedIds = blobTracker.getTrackedIds();

        if(trackedIds.size() >= 4){
            std::cout << "Position updating: " << '\n';
            for (const auto& trackedBlob : blobTracker.getTrackedBlobs()){
                std::cout << "Blob Center" << trackedBlob.center << '\n';
            }
        }
        writeTrackedIdsToFile(trackedIds, outFile);

        frame.copyTo(blobs);
        Mat mask = Mat::zeros(frame.size(), CV_8UC1);
        for (const auto& trackedBlob : blobTracker.getTrackedBlobs()) {
            circle(blobs, trackedBlob.center, trackedBlob.radius, trackedBlob.color, 2);
            putText(blobs, to_string(trackedBlob.id), trackedBlob.center, FONT_HERSHEY_SIMPLEX, 0.5, trackedBlob.color, 2);
            circle(mask, trackedBlob.center, trackedBlob.radius, Scalar(255), -1);
        }

        frame.copyTo(masked_blobs, mask);

        int width = 720;
        int height = 480;
        Mat resized_blobs, resized_mask;
        resize(blobs, resized_blobs, Size(width, height), INTER_LINEAR);
        resize(masked_blobs, resized_mask, Size(width, height), INTER_LINEAR);

        Mat combined;
        hconcat(resized_blobs, resized_mask, combined);

        imshow("Tracked Blobs and Masked Blobs", combined);
        if (waitKey(30) == 27) {
            break;
        }

        frameCount++;
    }

    outFile.close();
    return 0;
}