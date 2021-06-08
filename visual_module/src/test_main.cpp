#include <string>
#include <iostream>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include "visual_processing.h"
#include "path_smoothing.h"

using namespace std;
using namespace cv;

const string kWindowName = "Main Window in test_main";
bool path_set_planned = false;
LK_Tracker tracker(kWindowName);
ImgExtractor extractor(1);
PathSetTracker path_set_tracker;
vector<vector<Point2f>> path_set;

void ProcessImg(const sensor_msgs::ImageConstPtr& msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& execption_type) {
        ROS_ERROR("cv_bridge exception: %s", execption_type.what());
        return;
    }

    Mat cur_gray_img;
    vector<Point2f> projection_pts_on_path_set;
    cvtColor(cv_ptr->image, cur_gray_img, COLOR_BGR2GRAY); 
    
    tracker.Track(cv_ptr->image, cur_gray_img);   
    tracker.UpdateJd();
    cout << "Jd(6 x 2):\n" 
         << tracker.cur_Jd_ << '\n';
    extractor.Extract(cv_ptr->image);
    
    if (!path_set_planned && extractor.obs_extract_succeed_) {
        int pivot_idx = 0;
        float feedback_pts_radius = 10;
        vector<Point2f> initial_feedback_pts = tracker.GetFeedbackPoints(),
                        target_feedback_pts;
        target_feedback_pts = initial_feedback_pts;
        for (Point2f& pt : target_feedback_pts)
            pt += Point2f(200, 200);

        vector<PolygonObstacle> empty_obs;
        if (!initial_feedback_pts.empty()) {
            path_set = GeneratePathSet(initial_feedback_pts, target_feedback_pts, pivot_idx, feedback_pts_radius,
                                        empty_obs, cv_ptr->image);
            path_set_planned = !path_set.empty();
            if (path_set_planned)
                path_set_tracker = PathSetTracker(path_set);
        }
    }
    else if (path_set_planned) {
        projection_pts_on_path_set = path_set_tracker.ProjectPtsToPathSet(tracker.GetFeedbackPoints());
        for (auto& path : path_set) {
            for (int i = 0; i < int(path.size() - 1); i++)
                line(cv_ptr->image, path[i], path[i + 1], Scalar(120, 150, 120), 2);
        }
    }


    if (extractor.DO_extract_succeed_)
        drawContours(cv_ptr->image, extractor.DO_contours_, extractor.largest_DO_countor_idx_, Scalar(0, 255, 0), 2);
    if (extractor.obs_extract_succeed_)
        drawContours(cv_ptr->image, extractor.obs_contours_, 0, Scalar(0, 0, 255), 2);
    extractor.ProjectDOToObstacles();
    line(cv_ptr->image, extractor.DO_to_obs_projections_[0].first, 
        extractor.DO_to_obs_projections_[0].second, Scalar(0, 0, 255), 2);
    imshow(kWindowName, cv_ptr->image);
    waitKey(2);
}

int main(int argc, char** argv) {
    string ros_image_stream = "cameras/source_camera/image";
    namedWindow(kWindowName);
    
    ros::init(argc, argv, "test_main");
    ros::NodeHandle node_handle;
    image_transport::ImageTransport img_transport(node_handle);
    image_transport::Subscriber img_subscriber = img_transport.subscribe(ros_image_stream, 30, ProcessImg);
    ros::spin();

    return 0;
}