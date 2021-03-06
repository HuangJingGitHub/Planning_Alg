#ifndef PATH_SMOOTHING_INCLUDED
#define PATH_SMOOTHING_INCLUDED

#include <eigen3/Eigen/Dense>
#include "RRTStar_DOM_optimized.h"

using namespace Eigen;

vector<Point2f> QuadraticBSplineSmoothing(vector<RRTStarNode*>& node_path) {
    vector<Point2f> res;
    if (node_path.size() < 3) {
        cout << "No sufficient control points on the input path!\n"
             << "(At least 3 control points are needed for quadratic B-Spline smoothing.)\n";
        return res;
    }

    const float step_width = 0.05;   // For block() mathod needs constexpr as argument.
    const int step_num = 1 / step_width + 1,
              pts_num = node_path.size();

    Matrix3f coefficient_mat_1, coefficient_mat_2, coefficient_mat_3;
    MatrixXf parameter_var(3, step_num), 
             control_pts(2, pts_num);
    Matrix<float, 2, Dynamic> smoothed_path;

    coefficient_mat_1 << 1, -2, 1, -1.5, 2, 0, 0.5, 0, 0;
    coefficient_mat_2 << 0.5, -1, 0.5, -1, 1, 0.5, 0.5, 0, 0;
    coefficient_mat_3 << 0.5, -1, 0.5, -1.5, 1, 0.5, 1, 0, 0;
    
    for (int i = 0; i < step_num; i++) {
        float cur_var = step_width * i;
        parameter_var(0, i) = cur_var * cur_var;
        parameter_var(1, i) = cur_var;
        parameter_var(2, i) = 1;
    }
    for (int i = 0; i < pts_num; i++) {
        control_pts(0, i) = node_path[i]->pos.x;
        control_pts(1, i) = node_path[i]->pos.y;
    }

    MatrixXf cur_pts = control_pts.block<2, 3>(0, 0);
    MatrixXf cur_spline = cur_pts * coefficient_mat_1 * parameter_var;
    smoothed_path = cur_spline;

    for (int i = 1; i < pts_num - 2; i++) {
        cur_pts = control_pts.block<2, 3>(0, i);
        cur_spline = cur_pts * coefficient_mat_2 * parameter_var;
        smoothed_path.conservativeResize(2, smoothed_path.cols() + cur_spline.cols());
        smoothed_path.block<2, step_num>(0, smoothed_path.cols() - step_num) = cur_spline;
    }
    cur_pts = control_pts.block<2, 3>(0, pts_num - 3);
    cur_spline = cur_pts * coefficient_mat_3 * parameter_var;
    smoothed_path.conservativeResize(2, smoothed_path.cols() + cur_spline.cols());
    smoothed_path.block<2, step_num>(0, smoothed_path.cols() - step_num) = cur_spline;

/*     cout << "path smoothing:\n"
         << node_path.size() << ' '
         << smoothed_path.rows() << " x " << smoothed_path.cols() << '\n'
         << cur_spline.rows() << " x " << cur_spline.cols() << endl; */
    for (int i = 0; i < smoothed_path.cols(); i++)
        res.push_back(Point2f(smoothed_path(0, i), smoothed_path(1, i)));
    return res;
}


int SearchByDistance(vector<Point2f>& search_path, Point2f desired_pos) {
    int res_idx = 0;
    if (search_path.empty()) {
        cout << "An empty path is given." << endl;
        return res_idx;
    }

    Point2f trivial_sol = search_path.back();
    float ref_distance = norm(desired_pos - trivial_sol), min_distance_dif = FLT_MAX,
          cur_distance_dif;

    for (int i = search_path.size() - 2; i >= 0; i--) {
        cur_distance_dif = abs(norm(desired_pos - search_path[i]) - ref_distance);
        if (cur_distance_dif < min_distance_dif) {
            min_distance_dif = cur_distance_dif;
            res_idx = i;
        }
    }
    return res_idx;
}

vector<vector<Point2f>> GeneratePathSet(vector<Point2f>& initial_feedback_pts, 
                                        vector<Point2f>& target_feedback_pts, 
                                        int pivot_idx,
                                        float feedback_pts_radius,
                                        vector<PolygonObstacle>& obs,
                                        Mat source_img) {
    vector<vector<Point2f>> res_path_set(initial_feedback_pts.size());                                            
    RRTStarPlanner planner(initial_feedback_pts[pivot_idx], target_feedback_pts[pivot_idx],
                            obs);
    bool plan_success = planner.Plan(source_img, feedback_pts_radius, true);
    if (!plan_success) {
        cout << "Path planning failed!\n";
        return res_path_set;
    }

    vector<RRTStarNode*> pivot_path = planner.GetPath();
    vector<RRTStarNode*> sparse_pivot_path;
    for (int i = 0; i < pivot_path.size(); i += 2)
        sparse_pivot_path.push_back(pivot_path[i]);
    if (sparse_pivot_path.back() != pivot_path.back())
        sparse_pivot_path.push_back(pivot_path.back());
    vector<Point2f> smooth_povit_path = QuadraticBSplineSmoothing(sparse_pivot_path);
    res_path_set[pivot_idx] = smooth_povit_path;

    for (int i = 0; i < initial_feedback_pts.size(); i++) {
        if (i != pivot_idx) {
            Point2f cur_dif = initial_feedback_pts[i] - initial_feedback_pts[pivot_idx];
            res_path_set[i] = smooth_povit_path;
            for (Point2f& pt : res_path_set[i])
                pt += cur_dif;
            
            int truncation_idx = SearchByDistance(res_path_set[i], target_feedback_pts[i]);
            res_path_set[i].erase(res_path_set[i].begin() + truncation_idx, res_path_set[i].end());
            res_path_set[i].push_back(target_feedback_pts[i]);
        }
    }
    return res_path_set; 
}

vector<float>  GetLocalPathWidth2D( vector<Point2f>& path, 
                                    vector<PolygonObstacle>& obstacles,
                                    Size2f config_size = Size2f(640, 480)) {
    vector<float> local_path_width(path.size(), 0);
    vector<int> path_width_type(path.size(), 0);  // 0: two-side free, 1: one-side free, one-side obstacle
                                                  // 2: two-side obstacle
    float slope, x_max = config_size.width, y_max = config_size.height,
          cur_x, cur_y,
          x_intercept, y_intercept,
          x_at_y_max, y_at_x_max;

    for (int i = 1; i < path.size() - 1; i++) {
        cur_x = path[i].x;
        cur_y = path[i].y;
        slope = -(path[i + 1].x - path[i - 1].x) / (path[i + 1].y - path[i - 1].y);
        vector<Point2f> boundary_insection_pts;

        y_intercept = cur_y + slope * (0 - cur_x);
        if (0 <= y_intercept && y_intercept <= y_max)
            boundary_insection_pts.push_back(Point2f(0, y_intercept));
        x_intercept = cur_x - cur_y / slope;
        if (0 <= x_intercept && x_intercept <= x_max)
            boundary_insection_pts.push_back(Point2f(x_intercept, 0));
        x_at_y_max = cur_x + (y_max - cur_y) / slope;
        if (0 <= x_at_y_max && x_at_y_max <= x_max)
            boundary_insection_pts.push_back(Point2f(x_at_y_max, y_max));
        y_at_x_max = cur_x + slope * (x_max - cur_x);
        if (0 <= y_at_x_max && y_at_x_max <= y_max)
            boundary_insection_pts.push_back(Point2f(x_max, y_at_x_max));
        
        Point2f end_pt1 = boundary_insection_pts[0], end_pt2 = boundary_insection_pts[1],
                direction1 = end_pt1 - path[i], direiction2 = end_pt2 - path[i];
        vector<Point2f> obs_intersection_pts, direction1_pts, direction2_pts;
        for (PolygonObstacle& cur_obs : obstacles) {
            if (ObstacleFree(cur_obs, end_pt1, end_pt2))
                continue;
            Point2f cur_obs_intersection_pt = GetClosestIntersectionPt(cur_obs, end_pt1, end_pt2, path[i]);
            obs_intersection_pts.push_back(cur_obs_intersection_pt);
        }
        
        if (obs_intersection_pts.empty())
            local_path_width[i] = cv::norm(end_pt1 - end_pt2);
        else {
            for (Point2f cur_pt : obs_intersection_pts) {
                Point2f cur_vec = cur_pt - path[i];
                if (cur_vec.x * direction1.x + cur_vec.y * direction1.y > 0)
                    direction1_pts.push_back(cur_pt);
                else
                    direction2_pts.push_back(cur_pt);
            }
            if (!direction1_pts.empty()) {
                path_width_type[i]++;
                for (Point2f& cur_direction1_pt : direction1_pts) {
                    if (normSqr(cur_direction1_pt - path[i]) < normSqr(end_pt1 - path[i]))
                        end_pt1 = cur_direction1_pt;
                }
            }
            if (!direction2_pts.empty()) {
                path_width_type[i]++;
                for (Point2f& cur_direction2_pt : direction2_pts)
                    if (normSqr(cur_direction2_pt - path[i]) < normSqr(end_pt2 - path[i]))
                        end_pt2 = cur_direction2_pt;
            }
            local_path_width[i] = cv::norm(end_pt1 - end_pt2);
        }
    }

    if (local_path_width.size() >= 2) {
        local_path_width.front() = local_path_width[1];
        local_path_width.back() = local_path_width[local_path_width.size() - 2];
    }
    return local_path_width;
}

#endif