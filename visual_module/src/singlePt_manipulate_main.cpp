#include "ros/ros.h"
#include "arm_include/FvrRobotClient.hpp"
#include "arm_include/ConnectionManager.hpp"
#include "arm_include/ClientTestFunction.hpp"

#include "visual_module/visual_info_service_singlePt.h"

using namespace fvr;

Eigen::Matrix3f camera2base;
Eigen::Vector2f feedback_pt, target_pt,
                ee_pt, DO_to_obs_DO_pt, DO_to_obs_obs_pt,
                projection_on_path_pt;
Eigen::Matrix2f Jd;

void ProcessServece(const visual_module::visual_info_service_singlePt& srv) {
    feedback_pt(0, 0) = srv.response.feedback_pt[0];
    feedback_pt(1, 0) = srv.response.feedback_pt[1];
    target_pt(0, 0) = srv.response.target_pt[0];
    target_pt(1, 0) = srv.response.target_pt[1];
    ee_pt(0, 0) = srv.response.ee_pt[0];
    ee_pt(1, 0) = srv.response.ee_pt[1];

    for (int row = 0, cnt = 0; row < 2; row++)
        for (int col = 0; col < 2; col++) {
            Jd(row, col) = srv.response.Jd[cnt];
            cnt++;
        }
    
    DO_to_obs_DO_pt(0, 0) = srv.response.DO_to_obs_DO_pt[0];
    DO_to_obs_DO_pt(1, 0) = srv.response.DO_to_obs_DO_pt[1];
    DO_to_obs_obs_pt(0, 0) = srv.response.DO_to_obs_obs_pt[0];
    DO_to_obs_obs_pt(1, 0) = srv.response.DO_to_obs_obs_pt[1];
    projection_on_path_pt(0, 0) = srv.response.projection_on_path_pt[0];
    projection_on_path_pt(1, 0) = srv.response.projection_on_path_pt[1];
}


int main(int argc, char** argv) {
    camera2base <<  1, 0, 0,
                    0, 1, 0,
                    0, 0, 1;     
    std::shared_ptr<FvrRobotClient> robot = std::make_shared<FvrRobotClient>();
    
    const std::string server_address = "1";
    const std::string client_address = "2";
    ConnectionManager robot_connection(robot, server_address, client_address);
    
    ros::init(argc, argv, "service_client_test");
    ros::NodeHandle node_handle;
    ros::ServiceClient service_client = node_handle.serviceClient<visual_module::visual_info_service_singlePt>("visual_info_service_singlePt");
    visual_module::visual_info_service_singlePt srv;

    std::string start_flag;
    std::cout << "Press 1 to start experiment.\n";
    getline(std::cin, start_flag);
    if (start_flag != "1") {
        std::cout << "Exiting program.\n";
        return -1;
    }

    while (true) {
        if (!service_client.call(srv)) {
            std::cout << "Failed to call service.\n";
            return 1;
        }

        ProcessServece(srv);
    }
    return 0;
}