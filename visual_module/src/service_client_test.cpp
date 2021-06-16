#include "ros/ros.h"
#include "visual_module/visual_info_service_singlePt.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "service_client_test");
    ros::NodeHandle node_handle;
    ros::ServiceClient client = node_handle.serviceClient<visual_module::visual_info_service_singlePt>("visual_info_service_singlePt");

    visual_module::visual_info_service_singlePt srv;

    while (true) {
        if (client.call(srv)) {
            ROS_INFO("Succeed in geting response.\n");
            for (auto x : srv.response.Jd)
                ROS_INFO("%f ", x);
        }
        else {
            ROS_ERROR("Failed to call service.\n");
            return 1;
        }
    }

    return 0;
}