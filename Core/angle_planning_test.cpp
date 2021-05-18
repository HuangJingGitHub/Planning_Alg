#include <opencv2/imgproc.hpp>
#include "feature_geometry.h"
#include "RRTStar_DOM_debug.h"

int main(int argc, char** argv) {
    Mat backImg(Size(640, 480), CV_64FC3, Scalar(255, 255, 255));
    
    vector<float> test_angle_pos{200, 100, 100, 120, 300, 120};
    Angle3pts test_angle(test_angle_pos);

    vector<Point2f> vertices1{Point2f(310, 0), Point2f(330, 0), Point2f(330, 200), Point2f(310, 200)};
    PolygonObstacle obs1(vertices1);
    vector<PolygonObstacle> obstacles{obs1};

    vector<float> desired_s_0_pos{400, 100};
    vector<float> desired_config = test_angle.DetermineTarget(test_angle_pos, 60, desired_s_0_pos,
                                                                0, 0.05, obstacles, false);

    for (int i = 0; i < 3; i++) {
        circle(backImg, Point2f(test_angle.points_pos_(2 * i, 0), test_angle.points_pos_(2 * i + 1,0)),
               3, Scalar(255, 0, 0), -1);
        circle(backImg, Point2f(desired_config[2 * i], desired_config[2 * i + 1]), 3, 
                Scalar(0, 0, 255), -1);
        if (i != 0) {
            line(backImg, Point2f(test_angle.points_pos_(0, 0), test_angle.points_pos_(1,0)),
                 Point2f(test_angle.points_pos_(2 * i, 0), test_angle.points_pos_(2 * i + 1,0)),
                 Scalar(255, 0, 0), 2);
            line(backImg, Point2f(desired_config[0], desired_config[1]), 
                 Point2f(desired_config[2 * i], desired_config[2 * i + 1]), Scalar(0, 0, 255), 2);
        }
    }

    test_angle.Update(desired_config);
    int povit_idx = test_angle.GetPivotIndex();
    cout << "povit index: " << povit_idx << endl;

    Point2f povit_start(test_angle_pos[2 * povit_idx], test_angle_pos[2 * povit_idx + 1]),
            povit_target(desired_config[2 * povit_idx], desired_config[2 * povit_idx + 1]);
    RRTStarPlanner RRTStar_planer(povit_start, povit_target, obstacles, 20, 20, 10, Size2f(640, 480));
    bool planned = RRTStar_planer.Plan(backImg);

    rectangle(backImg, Point(310, 0), Point(330, 200), Scalar(0, 0, 0), 2);
    imshow("Target Configuration Determiantion", backImg);
    waitKey(0);
    return 0;
}