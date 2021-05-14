#include <math.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

class Angle3pts {
    // const int pos_DOFs_ = 6;
    /*Points storage order: s_0, s_1, s_2.*/
public:    
    Matrix<float, 6, 1> points_pos_;
    Matrix<float, 6, 1> gradient_;
    Vector2f v_1_;
    Vector2f v_2_;
    float angle_val_;   // deg
    
    Angle3pts(): points_pos_(ArrayXXf::Zero(6, 1)), gradient_(ArrayXXf::Zero(6, 1)), angle_val_(0) {}
    Angle3pts(vector<float> initial_pos) {
        if (initial_pos.size() != 6) {
            cout << "Invalid initial points coordinates. Custom initialization failed.\n"
                 << "The input vector should have a length of 6, but " << initial_pos.size() 
                 << " is given.\n";
            points_pos_ = ArrayXXf::Zero(6, 1);
        }
        else {
            for (int i = 0; i < 6; i++)
                points_pos_(i, 0) = initial_pos[i];
        }
        angle_val_ = 0;
    }

    void Update(vector<float> new_point_pos) {
        if (new_point_pos.size() != 6) {
            cout << "Fail to update the featuer information since input vector has an invalid dimension of "
                 << new_point_pos.size() << ". The length of the input vector should be 6.\n";
            return;
        }
        for (int i = 0; i < 6; i++)
            points_pos_(i, 0) = new_point_pos[i];
        
        float x_0 = points_pos_(0, 0), y_0 = points_pos_(1, 0),
              x_1 = points_pos_(2, 0), y_1 = points_pos_(3, 0),
              x_2 = points_pos_(4, 0), y_2 = points_pos_(5, 0);
        v_1_ = Vector2f(x_1 - x_0, y_1 - y_0);
        v_2_ = Vector2f(x_2 - x_0, y_2 - y_0);
        float t = v_1_.dot(v_2_) / (v_1_.norm()*v_2_.norm()),
              const_1 = -1 / sqrt(1 - t*t),
              const_2 = v_1_.dot(v_2_),
              n_1 = v_1_.norm(), n_2 = v_2_.norm();

        angle_val_ = acos(t) * 180 / M_PI;
        gradient_(0,0) = const_1 * ((2*x_0-x_1-x_2)*pow(n_1,2)*pow(n_2,2) - const_2*
                        ((x_0-x_1)*pow(n_2,2)+(x_0-x_2)*pow(n_1,2))) / (pow(n_1,3)*pow(n_2,3));
        gradient_(1,0) = const_1 * ((2*y_0-y_1-y_2)*pow(n_1,2)*pow(n_2,2) - const_2*
                        ((y_0-y_1)*pow(n_2,2)+(y_0-y_2)*pow(n_2,2))) / (pow(n_1,3)*pow(n_2,3));
        gradient_(2,0) = const_1 * ((x_2-x_0)*pow(n_1,2)*n_2 - (x_1-x_0)*const_2*n_2) / (pow(n_1,3)*pow(n_2,2));
        gradient_(3,0) = const_1 * ((y_2-y_0)*pow(n_1,2)*n_2 - (y_1-y_0)*const_2*n_2) / (pow(n_1,3)*pow(n_2,2));
        gradient_(4,0) = const_1 * ((x_1-x_0)*pow(n_2,2)*n_1 - (x_2-x_0)*const_2*n_1) / (pow(n_1,2)*pow(n_2,3));
        gradient_(5,0) = const_1 * ((y_1-y_0)*pow(n_2,2)*n_1 - (y_2-y_0)*const_2*n_1) / (pow(n_1,2)*pow(n_2,3));
    }

    int GetPivotIndex() {
        vector<pair<float, int>> gradient_square_to_index;
        for (int i = 0; i < 3; i++) {
            float gradient_square = pow(gradient_(2*i), 2) + pow(gradient_(2*i + 1), 2);
            gradient_square_to_index.push_back(pair<float, int>(gradient_square, i));
        }
        auto max_square_pair = *max_element(gradient_square_to_index.begin(), gradient_square_to_index.end());
        return max_square_pair.second;
    }
};