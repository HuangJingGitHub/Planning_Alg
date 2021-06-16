#include <string>
#include <iostream>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include "visual_processing.h"
#include "path_processing.h"
#include "visual_module/visual_info_service_singlePt.h"

using namespace std;
using namespace cv;

class VisualCore {
private:
    ros::NodeHandle node_handle_;
    ros::ServiceServer visual_info_service = node_handle_.advertiseService("visual_info_service_singlePt", 
                                                                    &VisualCore::GetVisualInfoService, this);
    image_transport::ImageTransport image_trans_;
    image_transport::Subscriber image_subscriber_;

    const string kWindowName = "Main Window in test_main";
    Mat cur_gray_img_;
    bool path_set_planned_ = false;
    LK_Tracker tracker_;
    ImgExtractor extractor_;
    PathSetTracker path_set_tracker_;
    vector<vector<Point2f>> path_set_;
    vector<vector<float>> path_local_width_set_;
    vector<Point2f> projection_pts_on_path_set_;
    vector<Point2f> target_feedback_pts_;

public:
    VisualCore(string ros_image_stream): image_trans_(node_handle_) {
        tracker_ = LK_Tracker(kWindowName);
        extractor_ = ImgExtractor(1, kWindowName);
        image_subscriber_ = image_trans_.subscribe(ros_image_stream, 30, &VisualCore::ProcessImg, this);
        namedWindow(kWindowName);
    }

    ~VisualCore() {
        destroyWindow(kWindowName);
    }

    void ProcessImg(const sensor_msgs::ImageConstPtr& msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& exception_type) {
            ROS_ERROR("cv_bridge exception: %s", exception_type.what());
            return;
        }
        cvtColor(cv_ptr->image, cur_gray_img_, COLOR_BGR2GRAY); 
        
        tracker_.Track(cv_ptr->image, cur_gray_img_);   
        tracker_.UpdateJd();
        extractor_.Extract(cv_ptr->image);
        // cout << "Jd(2 x 2):\n" 
        //     << tracker_.cur_Jd_ << '\n';
        
        vector<Point2f> initial_feedback_pts = tracker_.GetFeedbackPoints();
        if (!path_set_planned_ && (!initial_feedback_pts.empty()) && extractor_.obs_extract_succeed_) {
            int pivot_idx = 0;
            float feedback_pts_radius = 10;
            target_feedback_pts_ = initial_feedback_pts;
            for (Point2f& pt : target_feedback_pts_)
                pt += Point2f(200, 200);
            
            path_set_ = GeneratePathSet(initial_feedback_pts, target_feedback_pts_, pivot_idx, feedback_pts_radius,
                                        extractor_.obs_polygons_, cv_ptr->image);
            path_set_planned_ = !path_set_.empty();
            if (path_set_planned_) {
                path_set_tracker_ = PathSetTracker(path_set_);
                for (vector<Point2f>& cur_path : path_set_) {
                    vector<float> cur_path_local_width = GetLocalPathWidth2D(cur_path, extractor_.obs_polygons_);
                    path_local_width_set_.push_back(cur_path_local_width);
                }
            }
        }
        else if (path_set_planned_) {
            projection_pts_on_path_set_ = path_set_tracker_.ProjectPtsToPathSet(tracker_.GetFeedbackPoints());
            for (auto& path : path_set_) {
                for (int i = 0; i < int(path.size() - 1); i++)
                    line(cv_ptr->image, path[i], path[i + 1], Scalar(100, 200, 150), 2);
            }
        }

        if (extractor_.DO_extract_succeed_)
            drawContours(cv_ptr->image, extractor_.DO_contours_, extractor_.largest_DO_countor_idx_, Scalar(0, 255, 0), 2);
        if (extractor_.obs_extract_succeed_) {
            for (int i = 0; i < extractor_.obs_vertices_picked_.size() - 1; i++) 
                line(cv_ptr->image, extractor_.obs_vertices_picked_[i], extractor_.obs_vertices_picked_[i + 1],
                    Scalar(0, 0, 255), 2);
            line(cv_ptr->image, extractor_.obs_vertices_picked_.front(), extractor_.obs_vertices_picked_.back(),
                Scalar(0, 0, 255), 2);
        }
        extractor_.ProjectDOToObstacles();
        line(cv_ptr->image, extractor_.DO_to_obs_projections_[0].first, 
            extractor_.DO_to_obs_projections_[0].second, Scalar(0, 0, 255), 2);
        
        imshow(kWindowName, cv_ptr->image);
        waitKey(2);
    }

    bool GetVisualInfoService(  visual_module::visual_info_service_singlePt::Request &request,
                                visual_module::visual_info_service_singlePt::Response &response) {
        if (tracker_.points_[0].empty() || tracker_.ee_points_[0].empty() || 
            extractor_.DO_to_obs_projections_.empty() || projection_pts_on_path_set_.empty())
            return false;

        response.feedback_pt[0] = tracker_.points_[0][0].x;
        response.feedback_pt[1] = tracker_.points_[0][0].y;
        response.target_pt[0] = target_feedback_pts_[0].x;
        response.target_pt[1] = target_feedback_pts_[0].y;
        response.ee_pt[0] = tracker_.ee_points_[0][0].x;
        response.ee_pt[1] = tracker_.ee_points_[0][0].y;
        for (int row = 0, cnt = 0; row < 2; row++)
            for (int col = 0; col < 2; col++) {
                response.Jd[cnt] = tracker_.cur_Jd_(row, col);
                cnt++;
            }
        response.DO_to_obs_DO_pt[0] = extractor_.DO_to_obs_projections_[0].first.x;
        response.DO_to_obs_DO_pt[1] = extractor_.DO_to_obs_projections_[0].first.y;
        response.DO_to_obs_obs_pt[0] = extractor_.DO_to_obs_projections_[0].second.x;
        response.DO_to_obs_obs_pt[1] = extractor_.DO_to_obs_projections_[0].second.y;
        response.projection_on_path_pt[0] = projection_pts_on_path_set_[0].x;
        response.projection_on_path_pt[1] = projection_pts_on_path_set_[0].y;

        return true;
    }
};

int main(int argc, char** argv) {
    string ros_image_stream = "cameras/source_camera/image";
    
    ros::init(argc, argv, "visual_core");
    VisualCore visual_process_obj(ros_image_stream);
    ros::spin();
    return 0;
}