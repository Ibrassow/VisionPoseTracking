#ifndef FRAME_HPP
#define FRAME_HPP


#include <opencv2/opencv.hpp>



class Frame 
{

public:

    Frame();
    Frame(cv::Mat &raw_frame, int timestamp);

    void SetFrame(cv::Mat &raw_frame);
    cv::Mat& GetFrame();
    void GetFrame(cv::Mat &frame);

    void SetFeatures(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
    void GetFeatures(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const;

    const std::vector<cv::KeyPoint>& GetKeypoints() const;
    const cv::Mat& GetDescriptors() const;


    int GetNumberFeatures();


    cv::Mat& GetGrayFrame();

    



private:
    
    cv::Mat frame;
    cv::Mat gray;
    int timestamp;



    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    std::vector<cv::DMatch> inlier_matches;


};



void ShowMatchedFrames(Frame &prev_frame, Frame &frame, const std::vector<cv::DMatch>& matches);


#endif