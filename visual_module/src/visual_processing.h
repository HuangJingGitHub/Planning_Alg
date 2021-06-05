#ifndef VISUAL_PROCESSING_HEADER
#define VISUAL_PROCESSING_HEADER

#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace cv;

Point2f feedback_pt_picked;
Point2f ee_pt_picked;
bool add_remove_pt = false;
bool ee_add_remove_pt = false;  // ee: end-effector

static void onMouse(int event, int x, int y, int, void*) {
    if (event == EVENT_LBUTTONDOWN) {
        feedback_pt_picked = Point2f((float)x, (float)y);
        add_remove_pt = true;
    }
    if (event == EVENT_LBUTTONDBLCLK) {
        ee_pt_picked = Point2f((float)x, (float)y);
        ee_add_remove_pt = true;
    }    
}


class LK_Tracker {
public: 
    string window_to_track_;
    TermCriteria termiantion_criteria_;
    static const int points_num_ = 3;  // Determination of Jd size needs a constexpr argument.
    vector<Point2f> points_[2];
    vector<Point2f> ee_points_[2];
    Scalar points_color_;
    Scalar ee_points_color_;
    Mat pre_gary_img_;
    Mat next_gray_img_;

    Eigen::MatrixXf pre_Jd_;
    Eigen::MatrixXf cur_Jd_;
    float update_rate_;
    bool Jd_initialized_ = false;

    LK_Tracker() {}
    LK_Tracker(const string win_name) {
        window_to_track_ = win_name;
        termiantion_criteria_ = TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
        points_color_ = Scalar(255, 0, 0);
        ee_points_color_ = Scalar(0, 255, 0);

        update_rate_ = 0.15;
        pre_Jd_ = Eigen::Matrix<float, points_num_ * 2, 2>::Ones();
    }

    void Track(Mat& image, Mat& input_gray_img) {
        next_gray_img_ = input_gray_img;
        setMouseCallback(window_to_track_, onMouse, 0);
        if (add_remove_pt && points_[0].size() < (size_t) points_num_) {
            vector<Point2f> temp;
            temp.push_back(feedback_pt_picked);
            cornerSubPix(next_gray_img_, temp, Size(11, 11), Size(-1, -1), termiantion_criteria_);
            points_[0].push_back(temp[0]);
            add_remove_pt = false;
        }
        if (ee_add_remove_pt && ee_points_[0].empty()) {
            vector<Point2f> temp;
            temp.push_back(ee_pt_picked);
            cornerSubPix(next_gray_img_, temp, Size(11, 11), Size(-1, -1), termiantion_criteria_);
            ee_points_[0].push_back(temp[0]);
            ee_add_remove_pt = false;
        }

        if (!points_[0].empty())
            InvokeLK(pre_gary_img_, next_gray_img_, points_[0], points_[1], image, points_color_);      
        if (!ee_points_[0].empty())
            InvokeLK(pre_gary_img_, next_gray_img_, ee_points_[0], ee_points_[1], image, ee_points_color_);

        cv::swap(pre_gary_img_, next_gray_img_);
    } 


    void InvokeLK(Mat& pre_gray_img, Mat& next_gray_img, vector<Point2f>& pre_pts, vector<Point2f>& next_pts,
                Mat& display_img, Scalar& pt_color) {
        vector<uchar> status;
        vector<float> error;
        if (pre_gray_img.empty())
            next_gray_img.copyTo(pre_gray_img);

        calcOpticalFlowPyrLK(pre_gray_img, next_gray_img, pre_pts, next_pts, status, error, Size(11, 11),
                                3, termiantion_criteria_, 0, 0.001);
        size_t i, k;
        for (i = k = 0; i < next_pts.size(); i++) {
            if (add_remove_pt) {
                if (norm(feedback_pt_picked - next_pts[i]) <= 5) {
                    add_remove_pt = false;
                    continue;
                }
            }
            if (!status[i])
                continue;
            next_pts[k++] = next_pts[i];
            circle(display_img, next_pts[i], 3, pt_color, -1, 8);
        }
        next_pts.resize(k);   
        std::swap(pre_pts, next_pts);     
    }

    
    void UpdateJd() {
        if (!(points_[0].size() == points_num_ && points_[1].size() == points_num_)
            || !(ee_points_[0].size() == 1 && ee_points_[1].size() == 1)) {
                cout << "Unsuccessful tracking. No update of deformation Jacobian performaed.\n";
                return;
            }
        Eigen::Matrix<float, points_num_ * 2, 1> delta_points;
        Eigen::Matrix<float, 2, 1> delta_ee;
        // Note poits_[0] is current value, points_[1] stores previous value due to swap() in LK algorithm.
        for (int i = 0; i < points_num_; i++) {
            delta_points(2 * i, 0) = points_[0][i].x - points_[1][i].x;
            delta_points(2 * i + 1, 0) = points_[0][i].y - points_[1][i].y; 
        }
        delta_ee(0, 0) = ee_points_[0][0].x - ee_points_[1][0].x;
        delta_ee(1, 0) = ee_points_[0][0].y - ee_points_[1][0].y;

        if (delta_ee.norm() < 0.05) {
            cur_Jd_ = pre_Jd_;
            return;
        }

        cur_Jd_ = pre_Jd_ + update_rate_ * (delta_points - pre_Jd_*delta_ee) / delta_ee.squaredNorm() * delta_ee.transpose();
        pre_Jd_ = cur_Jd_;
    }
};


class ImgExtractor {
public:
    Mat original_HSV_img_;
    Scalar DO_HSV_low_;
    Scalar DO_HSV_high_;
    Scalar obs_HSV_low_;
    Scalar obs_HSV_high_;
    vector<vector<Point>> DO_contours_;
    vector<Point> DO_contour_;
    vector<vector<Point>> obs_contours_;
    int largest_DO_countor_idx_ = 0;
    int obs_num_preset_;
    bool obs_extracted_times_ = 0;
    bool DO_extract_succeed_ = false;
    bool obs_extract_succeed_ = false;
    

    ImgExtractor() {}
    ImgExtractor(int obs_num) {
        DO_HSV_low_ = Scalar(142, 96, 72);
        DO_HSV_high_ = Scalar(180, 255, 255);
        obs_HSV_low_ = Scalar(0, 0, 0);
        obs_HSV_high_ = Scalar(255, 255, 255);
        obs_num_preset_ = obs_num;
    }

    void Extract(Mat& image, Mat& HSV_img, Mat& gray, Mat& prevGray, int occlusion = 0) {
        original_HSV_img_ = HSV_img;
        if (original_HSV_img_.empty()) {
            cout << "Invalid image for extracting!\n";
            return;
        }

        Mat destination_img;
        inRange(original_HSV_img_, DO_HSV_low_, DO_HSV_high_, destination_img);
        Moments m_dst = moments(destination_img, true);
        //cout << m_dst.m00 <<" " << m_dst.m10 << " " << m_dst.m01 << "\n";
        if (m_dst.m00 < 5000) {
            cout << "No DO detected!\n";
            DO_extract_succeed_ = false;
        }
        else {
            findContours(destination_img, DO_contours_, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
            largest_DO_countor_idx_ = 0;
            for (int i = 1; i < DO_contours_.size(); i++) {
                largest_DO_countor_idx_ = (DO_contours_[i].size() > DO_contours_[largest_DO_countor_idx_].size())
                                            ? i : largest_DO_countor_idx_;
            }
            DO_contour_ = DO_contours_[largest_DO_countor_idx_];
            DO_extract_succeed_ = true;
        }

        if (occlusion != 0) {
            int second_largetst_countor_idx = (largest_DO_countor_idx_ == 0) ? 1 : 0;
            for (int i = 0; i < DO_contours_.size(); i++){
                second_largetst_countor_idx = (DO_contours_[i].size() > DO_contours_[second_largetst_countor_idx].size() && i != largest_DO_countor_idx_)
                                                ? i : second_largetst_countor_idx;
            }
            DO_contour_.insert(DO_contour_.end(), DO_contours_[second_largetst_countor_idx].begin(), DO_contours_[second_largetst_countor_idx].end());
            DO_contours_[largest_DO_countor_idx_] = DO_contour_; 
        }

        if (obs_extracted_times_ < 30) {
            inRange(original_HSV_img_, obs_HSV_low_, obs_HSV_high_, destination_img);
            m_dst = moments(destination_img, true);
            if (m_dst.m00 < 3000) {
                cout << "No obstacles detected!\n";
                obs_extract_succeed_ = false;
            }
            else {
                vector<vector<Point>> cur_obs_contours;
                findContours(destination_img, cur_obs_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
                if (cur_obs_contours.size() >= obs_num_preset_) {
                    vector<pair<int, int>> size_idx_pairs;
                    for (int i = 0; i < cur_obs_contours.size(); i++)
                        size_idx_pairs.push_back(pair<int, int>(cur_obs_contours[i].size(), i));
                    std::sort(size_idx_pairs.begin(), size_idx_pairs.end());
                    if (obs_contours_.empty()) {
                        for (int i = 0; i < obs_num_preset_; i++)
                            obs_contours_.push_back(cur_obs_contours[size_idx_pairs[i].second]);
                    }
                    else {
                        for (int i = 0; i < obs_num_preset_; i++) {
                            obs_contours_[i] = cur_obs_contours[size_idx_pairs[i].second];
                            /*Averaging operation is hard to define (find correspinding point pair, 
                            dimensional difference, etc). Instead, use the updated data after 
                            sufficient iterations.*/
                            obs_extract_succeed_ = true;
                        }
                    }
                }
            }
        }

    }
};

#endif