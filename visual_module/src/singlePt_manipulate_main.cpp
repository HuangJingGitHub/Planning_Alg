#include "ros/ros.h"
#include <eigen3/Eigen/QR>
#include "arm_include/FvrRobotClient.hpp"
#include "arm_include/ConnectionManager.hpp"
#include "arm_include/ClientTestFunction.hpp"
#include "visual_module/visual_info_service_singlePt.h"

using namespace fvr;

int ONLINE_MOVE_MODE = 8;
float motion_amplitude = 0.01;

Eigen::Matrix3f camera2base;
Eigen::Vector2f feedback_pt, target_pt,
                ee_pt, DO_to_obs_DO_pt, DO_to_obs_obs_pt,
                projection_on_path_pt,
                error_pt, ee_velocity_image;
Eigen::Vector3f ee_velocity_image_3D = Eigen::Array3f::Zero(),
                ee_velocity_3D;
Eigen::Matrix2f Jd = Eigen::Matrix2f::Identity();

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
    
    error_pt = projection_on_path_pt - target_pt;
    ee_velocity_image = -Jd.inverse() * error_pt;
    ee_velocity_image_3D(0, 0) = ee_velocity_image(0, 0);
    ee_velocity_image_3D(1, 0) = ee_velocity_image(1, 0);

    std::cout << "Jd: \n" << Jd 
            << "\nJd^-1:\n" << Jd.inverse() << '\n';
}


int main(int argc, char** argv) {
    camera2base <<  1, 0, 0,
                    0, 1, 0,
                    0, 0, 1;
    std::shared_ptr<FvrRobotClient> robot = std::make_shared<FvrRobotClient>();
    
    const std::string server_address = "1";
    const std::string client_address = "2";
    ConnectionManager robot_connection(robot, server_address, client_address);
    std::thread connection([&]() {
        while (true) {
            if (robot_connection.run() != true)
                return
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    });

    ClientTestFunction client_tester(robot);
    // client_tester.printTestCase();

    ros::init(argc, argv, "manipulate_singelePt_main");
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
        ee_velocity_3D = camera2base * ee_velocity_image_3D;
        ee_velocity_3D = motion_amplitude / ee_velocity_3D.norm() * ee_velocity_3D;

        if (robot_connection.robotConnected() == false) {
            std::cout << "Fial to connect to robot server!\n";
            continue;
        }
        if (client_tester(ONLINE_MOVE_MODE, ee_velocity_3D) != true) {
            std::cout << "Fail to execute the command!\n";
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return 0;
}