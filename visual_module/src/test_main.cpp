#include <string>
#include <iostream>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include "visual_processing.h"

using namespace std;
using namespace cv;
const string kWindowName = "Main Window in test_main";
LK_Tracker tracker(kWindowName);

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
    cvtColor(cv_ptr->image, cur_gray_img, COLOR_BGR2GRAY); 
    tracker.Track(cv_ptr->image, cur_gray_img);   
    tracker.UpdateJd();
    cout << "Jd(6 x 2):\n" 
         << tracker.cur_Jd_ << '\n';

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