#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include <unistd.h>    

#include "camera.hpp"
#include "extractor.hpp"


int main (int argc, char** argv)
{


    Camera cam(0);
    cam.TurnOn();
    cam.LoadIntrinsics("../params/cam.yaml");

    
    FeatureExtractor extractor;
    extractor.SetMaxOrbFeatures(1000);

    Frame current_frame;
    Frame prev_frame;

    std::vector<cv::DMatch> matches;


    std::vector<cv::Point2f> prev_points;
    std::vector<cv::Point2f> curr_points;

    cv::Mat rotation, translation;


    cv::Mat intrinsics;
    cam.GetIntrinsics(intrinsics);

    bool init = true;
    double  prev_frame_time;
    double current_frame_time = cv::getTickCount();
    double elapsed_seconds;

    int nb_matches = 0;
    int nb_features = 0;

    sleep(1);


    while (true) 
    {


        cam.CaptureFrame(current_frame);

        
        extractor.ExtractFeatures(current_frame);
        nb_features = current_frame.GetNumberFeatures();
        std::cout << "Number of keypoints: " << nb_features << std::endl;
        if (nb_features < 5) {
            std::cout << "Not enough features" << std::endl;
            continue;
        }



        if (init) {
            prev_frame = Frame(current_frame);

            prev_frame_time = cv::getTickCount();
            current_frame_time = cv::getTickCount();


            init = false;
            continue;
        }



        // Descriptor matching
        nb_matches = extractor.MatchFeatures(prev_frame, current_frame, matches);
        std::cout << "Number of matches: " << nb_matches << std::endl;
        if (nb_matches < 5) {
            std::cout << "Not enough matches" << std::endl;
            continue;
        }
        extractor.RetrieveMatches(prev_frame, current_frame, matches, prev_points, curr_points);

        std::vector<int> inliers_idx;
        extractor.EstimateRelativePoseWithEssential(prev_points, curr_points, intrinsics, inliers_idx);
        


        // Print the rotation and translation
        extractor.GetRotation(rotation);
        extractor.GetTranslation(translation);
        std::cout << "Rotation: " << rotation << std::endl;
        std::cout << "Translation: " << translation << std::endl;

        std::vector<cv::DMatch> inlier_matches;
        extractor.GetInliersMatches(matches, inliers_idx, inlier_matches);
        ShowMatchedFrames(prev_frame, current_frame, inlier_matches);

        


        prev_points.clear();
        curr_points.clear();

        prev_frame = Frame(current_frame);


        current_frame_time = cv::getTickCount();
        elapsed_seconds = (current_frame_time - prev_frame_time) / cv::getTickFrequency();
        std::cout << "FPS: " <<  1.0 / elapsed_seconds << std::endl;
        prev_frame_time = current_frame_time;


        // Exit the loop if 'q' is pressed
        if (cv::waitKey(1) == 'q') {
            break;
        }

    }


    // Release the camera and close all windows
    cam.TurnOff();
    cv::destroyAllWindows();

    return 0;
}