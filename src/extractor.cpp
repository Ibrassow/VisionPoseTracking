#include "extractor.hpp"


FeatureExtractor::FeatureExtractor()
{
    orb = cv::ORB::create();
    bf_matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
}



void FeatureExtractor::SetMaxOrbFeatures(int max_features)
{
    orb->setMaxFeatures(max_features);
}


bool FeatureExtractor::ExtractFeatures(Frame &frame)
{

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::Mat gray;

    
    //orb->detect(gray, keypoints);
    //orb->compute(gray, keypoints, descriptors);

    orb->detectAndCompute(frame.GetGrayFrame(), cv::Mat(), keypoints, descriptors);

    frame.SetFeatures(keypoints, descriptors);

    if (keypoints.size() == 0) {
        return false;
    }

    std::cout << "Extracted features" << std::endl;
    return true;
}

int FeatureExtractor::MatchFeatures(const Frame &prev_frame, const Frame &frame, std::vector<cv::DMatch>& matches, bool sorted, int top)
{   
    auto d1 = prev_frame.GetDescriptors();
    auto d2 = frame.GetDescriptors();
    // Watch out - order matters, especially for drawing
    bf_matcher->match(prev_frame.GetDescriptors(), frame.GetDescriptors(), matches);

    if (sorted) {
        std::sort(matches.begin(), matches.end());
    }

    int n = matches.size();
    if (n > top) {
        matches.erase(matches.begin() + top, matches.end());
    }
    

    return matches.size();

}

void FeatureExtractor::RetrieveMatches(
    const Frame &prev_frame, 
    const Frame &frame, 
    std::vector<cv::DMatch>& matches,
    std::vector<cv::Point2f>& prev_points,
    std::vector<cv::Point2f>& curr_points
    )
{
    
    prev_points.clear();
    curr_points.clear();
    for (auto& match : matches) {
        prev_points.push_back((prev_frame.GetKeypoints())[match.queryIdx].pt);
        curr_points.push_back((frame.GetKeypoints())[match.trainIdx].pt);
    }
}


bool FeatureExtractor::EstimateRelativePoseWithEssential(
    const std::vector<cv::Point2f>& points_img1, 
    const std::vector<cv::Point2f>& points_img2, 
    const cv::Mat &K, 
    std::vector<int>& inliers_idx
    )
{
    try
    {
        inliers_idx.clear();
                                    
        cv::Point2f principal_point(K.at<double>(0, 2), K.at<double>(1, 2));
        double focal_length = (K.at<double>(0, 0) + K.at<double>(1, 1)) / 2; 

        cv::Mat inliers_mask; 
        essential_matrix = cv::findEssentialMat(points_img1, points_img2, focal_length, principal_point, cv::RANSAC, 0.999, 1.0, inliers_mask);
        essential_matrix /= essential_matrix.at<double>(2, 2);

        for (int i = 0; i < inliers_mask.rows; i++)
        {
            if ((int)inliers_mask.at<unsigned char>(i, 0) == 1)
            {
                inliers_idx.push_back(i);
            }
        }

        recoverPose(essential_matrix, points_img1, points_img2, rotation, translation, focal_length, principal_point, inliers_mask);
        double norm = sqrt(translation.at<double>(0, 0) * translation.at<double>(0, 0) + translation.at<double>(1, 0) * translation.at<double>(1, 0) +
                    translation.at<double>(2, 0) * translation.at<double>(2, 0));
        translation /= norm;

        return true;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return false;
    }
    



}

void FeatureExtractor::GetInliersMatches(std::vector<cv::DMatch>& matches, std::vector<int>& inliers_idx, std::vector<cv::DMatch>& inliers_matches)
{
    inliers_matches.clear();
    for (int idx : inliers_idx) {
        const cv::DMatch &m = matches[idx];
        inliers_matches.push_back(
        cv::DMatch(m.queryIdx, m.trainIdx, m.distance));
    }
}

void FeatureExtractor::GetRotation(cv::Mat &rotation)
{
    rotation = this->rotation.clone();
}

void FeatureExtractor::GetTranslation(cv::Mat &translation)
{
    translation = this->translation.clone();
}