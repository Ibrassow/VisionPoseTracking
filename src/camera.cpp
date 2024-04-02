#include <iostream>
#include "camera.hpp"

Camera::Camera(int id)
: 
cam_id(id),
is_camera_on(false)
{}


void Camera::TurnOn()
{
    try {

        cap.open(cam_id);
        // Check if the camera is opened successfully
        if (!cap.isOpened()) {
            throw std::runtime_error("Unable to open the camera");
        }
        is_camera_on = true;

    } catch (const std::exception& e) {
        std::cout << "Exception occurred: " << e.what() << std::endl;
    }

}

void Camera::TurnOff()
{
    try {
        if (is_camera_on) {
            cap.release();
            is_camera_on = false;
        }
    } catch (const std::exception& e) {
        std::cout << "Exception occurred: " << e.what() << std::endl;
    }
}


Camera::~Camera()
{
    TurnOff();
}

void Camera::CaptureFrame(Frame& frame)
{
    
    try {
        if (!is_camera_on) {
            throw std::runtime_error("Camera is not turned on");
        }

        cv::Mat img, und_img;
        cap.read(img);

        undistort(img, und_img, intrinsics, distortion_parameters);

        if (img.empty()) {
            std::cout << "Unable to read frame from the camera" << std::endl;
            // TODO - high-level must check
        }

        frame = Frame(und_img, 0);


    } catch (const std::exception& e) {
        std::cout << "Exception occurred: " << e.what() << std::endl;
    }
}


bool Camera::LoadIntrinsics(const std::string& filename)
{
    try {
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        
        if (!fs.isOpened()) {
            throw std::runtime_error("Failed to open the YAML file");
        }

        fs["intrinsics"] >> intrinsics;
        fs["distortion_coefficients"] >> distortion_parameters;

        fs.release();
        return true;

    } catch (const std::exception& e) {
        std::cout << "Exception occurred: " << e.what() << std::endl;
        return false;
    }
}

void Camera::GetIntrinsics(cv::Mat& intrinsics)
{
    intrinsics = this->intrinsics.clone();
}