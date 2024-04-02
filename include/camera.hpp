#ifndef CAMERA_HPP
#define CAMERA_HPP


#include <opencv2/opencv.hpp>
#include<unistd.h>  
#include <string>

#include "frame.hpp"


class Camera
{

public:
    Camera(int id);
    ~Camera();



    void CaptureFrame(Frame& frame);

    bool LoadIntrinsics(const std::string& filename);

    void GetIntrinsics(cv::Mat& intrinsics);



    void TurnOn();


    void TurnOff();

    //void Display();


    
private:
    int cam_id;
    cv::VideoCapture cap;
    bool is_camera_on;

    cv::Mat intrinsics;
    cv::Mat distortion_parameters;


};

#endif // CAMERA_HPP