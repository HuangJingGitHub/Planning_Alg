#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <random>
#include <chrono>
#include <vector>
#include <queue>
#include <algorithm>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "Obstacles.h"

using namespace cv;
using namespace std;



class RRTStarPlanner {
public:
    Point2f start_pos_;
    Point2f target_pos_;
    vector<PolyObstacle> obstacles_;
    float step_len_;
    float error_dis_;
    float radius_;
    Size2f config_size_;
    int MAX_GRAPH_SIZE = 2500;
    int CUR_GRAPH_SIZE = 0;
    bool plan_scuess_ = false;
    vector<Point2f> graph_nodes_;
    vector<int> graph_parents_;
    vector<float> graph_costs_;
    int target_pos_parent_; 

    RRTStarPlanner() {}
    RRTStarPlanner(Point2f start, Point2f target, vector<PolyObstacle> obs, float step_len = 10, 
                   float radius = 10, float error_dis = 10,
                   Size2f config_size = Size2f(640, 480)): 
        start_pos_(start), 
        target_pos_(target), 
        obstacles_(obs),
        step_len_(step_len), 
        radius_(radius),
        error_dis_(error_dis),
        config_size_(config_size) {
        CUR_GRAPH_SIZE++;
        graph_parents_ = vector<int>(MAX_GRAPH_SIZE, 0);
        graph_costs_ = vector<float>(MAX_GRAPH_SIZE, 0);
        graph_nodes_.push_back(start_pos_);
    }
    ~RRTStarPlanner() {};

    bool Plan(Mat source_img) {        
        plan_scuess_ = false;

        srand(time(NULL));
        float div_width = RAND_MAX / config_size_.width,
              div_height = RAND_MAX / config_size_.height,
              min_cost = 1e8;
        /* unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        default_random_engine generator(seed);
        uniform_real_distribution<float>    distribution_x(0.0, (float) config_size_.width),
                                            distribution_y(0.0, (float) config_size_.height);*/
        Point2f rand_pos = Point2f(0, 0);
        
        while (CUR_GRAPH_SIZE < MAX_GRAPH_SIZE) {
            rand_pos.x = rand() / div_width;
            rand_pos.y = rand() / div_height;
            // rand_pos.x = distribution_x(generator);
            // rand_pos.y = distribution_y(generator);
            if (CUR_GRAPH_SIZE % 100 == 0) {
                rand_pos = target_pos_;
                CUR_GRAPH_SIZE++;
            }

            int nearest_idx = NearestNodeIdx(rand_pos);
            if (norm(rand_pos - graph_nodes_[nearest_idx]) < 2 && rand_pos != target_pos_)
                continue;
            Point2f new_node = AddNewNode(nearest_idx, rand_pos);
            if (PathObstacleFree(graph_nodes_[nearest_idx], new_node)) {
                    rewire(nearest_idx, new_node, source_img);
                if (norm(new_node - target_pos_) <= error_dis_) {
                    if (graph_costs_[graph_nodes_.size() - 1] + norm(target_pos_ - new_node) < min_cost){
                        target_pos_parent_ = graph_nodes_.size() - 1;
                        min_cost = graph_costs_[graph_nodes_.size() - 1] + norm(target_pos_ - new_node);
                    }
                    plan_scuess_ = true;
                }
                CUR_GRAPH_SIZE++;
                circle(source_img, new_node, 2, Scalar(0, 255, 0), -1);
                // circle(source_img, rand_pos, 10, Scalar(0, 0, 255), -1);
            }
            rectangle(source_img, Point(310, 0), Point(330, 200), Scalar(0, 0, 0), 2);
            rectangle(source_img, Point(310, 280), Point(330, 480), Scalar(0, 0, 0), 2);
            rectangle(source_img, Point(500, 280), Point(520, 480), Scalar(0, 0, 0), 2);
            circle(source_img, start_pos_, 5, Scalar(255,0,0), -1);
            circle(source_img, target_pos_, 5, Scalar(255,0,0), -1);
            // line(source_img, rand_pos, new_node, Scalar(0, 0, 255), 1);
            imshow("RRT* path planning", source_img);
            waitKey(2);
            cout << "-->CUR_GRAPH_SIZE " << graph_nodes_.size() << '\n';
        }
        if (!plan_scuess_)
            cout << "MAX_GRAPH_SIZE: " << MAX_GRAPH_SIZE << " is achieved with no path founded.\n";
        else
            cout << "Path found with cost: " << min_cost
                 << "\nOptimal cost: " << norm(target_pos_ - start_pos_)
                 << '\n';   
        return plan_scuess_;
    }

    int NearestNodeIdx(Point2f rand_node) {
        int res = 0;
        float min_dis = norm(rand_node - graph_nodes_[0]), cur_dis;
        for (int i = 1; i < graph_nodes_.size(); i++) {
            cur_dis = norm(rand_node - graph_nodes_[i]);
            if (cur_dis < min_dis) {
                res = i;
                min_dis = cur_dis;
            }
        }
        return res;
    }

    Point2f AddNewNode(int nearest_idx, Point2f rand_pos) {
        Point2f nearest_node = graph_nodes_[nearest_idx],
                direction = (rand_pos - nearest_node) / norm((rand_pos - nearest_node)),
                new_pos = nearest_node + direction * step_len_;
        // new_pos = rand_pos;
        return new_pos;
    }

    void rewire(int nearest_idx, Point2f new_node, Mat source_img) {
        float gamma_star = 1600,
              gamma = gamma_star * sqrt(log10(CUR_GRAPH_SIZE) * 3.32 / CUR_GRAPH_SIZE),
              radius_alg = min(gamma, step_len_);

        vector<int> near_set;
        for (int i = 0; i < graph_nodes_.size(); i++)
            if (norm(new_node - graph_nodes_[i]) < radius_)
                near_set.push_back(i);

        // find minimal cost path
        int min_cost_idx = nearest_idx;
        float min_cost = graph_costs_[nearest_idx] + norm(new_node - graph_nodes_[nearest_idx]);
        for (auto near_idx : near_set) {
            Point2f near_node = graph_nodes_[near_idx];
            float cur_cost = graph_costs_[near_idx] + norm(new_node - near_node);
            if (cur_cost < min_cost && PathObstacleFree(near_node, new_node)) {
                min_cost_idx = near_idx;
                min_cost = cur_cost;
            }
        }

        graph_nodes_.push_back(new_node);
        int new_node_idx = graph_nodes_.size() - 1;
        graph_parents_[new_node_idx] = min_cost_idx;
        graph_costs_[new_node_idx] = min_cost;
        line(source_img, graph_nodes_[min_cost_idx], new_node, Scalar(0, 0, 255), 1.5);

        for (auto near_idx : near_set) {
            if (graph_costs_[new_node_idx] + norm(new_node -graph_nodes_[near_idx]) < graph_costs_[near_idx] 
                && PathObstacleFree(graph_nodes_[near_idx], new_node)) {
                    graph_parents_[near_idx] = new_node_idx;
            }
        }

    }

    bool PathObstacleFree(Point2f near_node, Point2f new_node) {
        for (auto& obs : obstacles_)
            if (!ObstacleFree(obs, near_node, new_node))
                return false;
        return true;
    }
    
    vector<Point2f> GetPath() {
        vector<Point2f> res{target_pos_};
        if (!plan_scuess_) {
            cout << "No valid path available.\n";
            return res;
        }
        int idx = target_pos_parent_;
        while (idx != 0) {
            res.push_back(graph_nodes_[idx]);
            idx = graph_parents_[idx];
        }
        res.push_back(start_pos_);
        reverse(res.begin(), res.end());
        return res;   
    }    
};