#include <string>
#include <vector>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

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
    int max_count_;
    vector<Point2f> points_[2];
    vector<Point2f> ee_points_[2];
    Scalar points_color_;
    Scalar ee_points_color_;
    Mat original_img_;
    Mat pre_gary_img_;
    Mat next_gray_img_;

    LK_Tracker() {}
    LK_Tracker(const string win_name) {
        window_to_track_ = win_name;
        termiantion_criteria_ = TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
        max_count_ = 3;
        points_color_ = Scalar(255, 0, 0);
        ee_points_color_ = Scalar(0, 255, 0);
    }

    void Track(Mat& image, Mat& input_gray_img) {
        original_img_ = image;  // Just create a matrix header to the original image.
        next_gray_img_ = input_gray_img;
        setMouseCallback(window_to_track_, onMouse, 0);
        if (add_remove_pt && points_[0].size() < (size_t) max_count_) {
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
};


class ImgExtractor {
public:
    Scalar DO_HSV_low_;
    Scalar DO_HSV_high_;
    Mat original_HSV_img_;
    vector<vector<Point>> DO_contours_;
    vector<Point> DO_contour_; 
    int largest_DO_countor_idx_ = 0;
    bool DO_extract_succeed_ = false;
    bool extract_succeed_ = false;
    
    ImgExtractor() {
        DO_HSV_low_ = Scalar(142, 96, 72);
        DO_HSV_high_ = Scalar(180, 255, 255);
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
            cout << "No DO Detected!\n";
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
        extract_succeed_ = DO_extract_succeed_;
    }
};