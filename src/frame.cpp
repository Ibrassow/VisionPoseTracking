#include "frame.hpp"



Frame::Frame()
{
}


Frame::Frame(cv::Mat &raw_frame, int timestamp)
:
frame(raw_frame),
timestamp(timestamp)
{

    cv::cvtColor(raw_frame, gray, cv::COLOR_BGR2GRAY);
}


void Frame::SetFrame(cv::Mat &raw_frame)
{
    frame = raw_frame;
}

cv::Mat& Frame::GetFrame()
{
    return frame;
}

void Frame::GetFrame(cv::Mat &out)
{
    out = frame;
}


void Frame::SetFeatures(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    this->keypoints = keypoints;
    this->descriptors = descriptors;
}

void Frame::GetFeatures(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const
{
    keypoints = this->keypoints;
    descriptors = this->descriptors;
}


const std::vector<cv::KeyPoint>& Frame::GetKeypoints() const
{
    return keypoints;
}

const cv::Mat& Frame::GetDescriptors() const
{
    return descriptors;
}


int Frame::GetNumberFeatures()
{
    return keypoints.size();
}


cv::Mat& Frame::GetGrayFrame()
{
    return gray;
}



void ShowMatchedFrames(Frame &prev_frame, Frame &frame, const std::vector<cv::DMatch>& matches)
{
    cv::Mat prev_frame_image = prev_frame.GetFrame();
    cv::Mat frame_image = frame.GetFrame();

    cv::Mat out;

    cv::drawMatches(prev_frame_image, prev_frame.GetKeypoints(), frame_image, frame.GetKeypoints(), matches, out, cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0));
    cv::imshow("Matches", out);

}