#include "opencv2/imgproc.hpp"
#include "path_smoothing.h"
#include "feature_geometry.h"

int main(int argc, char** argv) {
    vector<RRTStarNode*> nodePath;
    vector<Point2f> res;

    vector<int> sample_x{1,2,4,5}, sample_y{1,5,2,3};
    for (int i = 0; i < 4; i++) {
        Point2f newPos = Point2f(sample_x[i], sample_y[i]);
        RRTStarNode *newNode = new RRTStarNode(newPos);
        nodePath.push_back(newNode);
    }
    
    res = QuadraticBSplineSmoothing(nodePath);

    vector<float> angle_pos{0, 0, 1, 0, 0, 1};
    Angle3pts angle;
    angle.Update(angle_pos);
    
    cout << angle.angle_val_ << " deg\n"
         << angle.gradient_ 
         << "\npivot: "
         << angle.GetPivotIndex()
         << endl;
    
    return 0;
}