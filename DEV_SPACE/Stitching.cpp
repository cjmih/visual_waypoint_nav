#include <iostream>
#include <fstream>
#include <glob.h> 
#include <string.h>
#include <vector>
#include <stdexcept>
#include <string>
#include <sstream>
// Include header files from OpenCV directory
// required to stitch images.
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"
#include <chrono>

using namespace std;
using namespace cv;
 
// Define mode for stitching as panorama 
// (One out of many functions of Stitcher)
Stitcher::Mode mode = Stitcher::PANORAMA;

std::vector<Mat> get_img_vec(const string& file_path){
    vector<cv::String> fn;
    glob(file_path, fn, false);
    // getting a vector of images
    vector<Mat> imgs;
    size_t count = fn.size(); //number of png files in images folder
    cout << "Count: " << count << endl; 
    for (size_t i=0; i<count; i++){
        imgs.push_back(imread(fn[i]));
        //cout << "Size: " << i << endl;
    }

    return imgs;
}

int main(int argc, char* argv[])
{


    
    // Array for pictures
    //vector<cv::String> fn;
    // using glob to parse the folder for a list of filenames
    //glob("/home/connor/ae295/OldImages", fn, false);
    string file_path = "/home/connor/AirSim/capture_map_images/sample";
    std::vector<Mat> imgs = get_img_vec(file_path);
    // Define object to store the stitched image
    Mat pano;
     
    // Create a Stitcher class object with mode panoroma
    Ptr<Stitcher> stitcher = Stitcher::create(mode);
    cout << "Starting Stitcher" << endl; 
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    // Command to stitch all the images present in the image array
    Stitcher::Status status = stitcher->stitch(imgs, pano);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;
 
    if (status != Stitcher::OK)
    {
        // Check if images could not be stitched
        // status is OK if images are stitched successfully
        cout << "Can't stitch images\n";
        return -1;
    }
    int width = 1440; 
    int height = 1440; 
    Mat resized_down;
    resize(pano, resized_down, Size(width, height), INTER_LINEAR);
    // Store a new image stitched from the given 
    //set of images as "result.jpg"
    //imwrite("result.jpg", resized_down);

    //filtering and bluring image for detection
    Mat img_gray;
    cvtColor(resized_down, img_gray, COLOR_BGRA2GRAY);

    // Mat img_blur;
    // GaussianBlur(img_gray, img_blur, Size(3,3), 0);

    // Mat edges;
    // int low_thresh = 100;
    // int high_thresh = 200;
    // Canny(img_gray, edges, low_thresh, high_thresh, 3, false);
    imwrite("results.jpg", resized_down);  
    // Show the result
    imshow("Result", resized_down);
    
    imshow("Gray", img_gray);

    // b nb-nn  imshow("Blur", img_blur);

    //imshow("Canny", edges);
     
    waitKey(0);
    return 0;
}