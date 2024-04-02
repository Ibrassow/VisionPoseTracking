#ifndef FeatureExtractor_HPP
#define FeatureExtractor_HPP

#include <vector>

#include <opencv2/opencv.hpp>

#include "frame.hpp"


class FeatureExtractor
{

    public:
        FeatureExtractor();


        void SetMaxOrbFeatures(int max_features);

        bool ExtractFeatures(Frame &frame);

        int MatchFeatures(const Frame &prev_frame, const Frame &frame, std::vector<cv::DMatch>& matches, bool sorted = true, int top = 1000);

        void RetrieveMatches(const Frame &prev_frame, const Frame &frame, std::vector<cv::DMatch>& matches, std::vector<cv::Point2f>& prev_points, std::vector<cv::Point2f>& curr_points);


        bool EstimateRelativePoseWithEssential(const std::vector<cv::Point2f>& points_img1, const std::vector<cv::Point2f>& points_img2, const cv::Mat &intrinsics, std::vector<int>& inliers_idx);

        void GetInliersMatches(std::vector<cv::DMatch>& matches, std::vector<int>& inliers_idx, std::vector<cv::DMatch>& inliers_matches);

        void GetRotation(cv::Mat &rotation);
        void GetTranslation(cv::Mat &translation);


    private:

        cv::Ptr<cv::ORB> orb;
        cv::Ptr<cv::BFMatcher> bf_matcher;

        std::vector<cv::DMatch> current_matches;
        cv::Mat essential_matrix;


        cv::Mat rotation;
        cv::Mat translation;


};




#endif // FeatureExtractor_HPP