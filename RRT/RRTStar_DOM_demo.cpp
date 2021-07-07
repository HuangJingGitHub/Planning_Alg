#include "opencv2/videoio.hpp"
#include "path_smoothing.h"

static void help() {
    // print a welcome message, and the OpenCV version
    cout << "\nThis is a demo of RRT* algorithm in image space,\n"
            "Using OpenCV version " << CV_VERSION << endl;
    cout << "\nIt uses camera input by default, but you can provide a path to video as an argument.\n";
    cout << "\nHot keys:\n"
            "\tESC - quit the program\n"
            "\tc - delete all the points\n"
            << endl;
}


Point2f point;
bool addRemovePt = false;
bool saveImg = false;
static void onMouse(int event, int x, int y, int /*flags*/, void* /*param*/ ) {
    if(event == EVENT_LBUTTONDOWN ) {
        point = Point2f((float)x, (float)y);
        addRemovePt = true;
    }
    else if (event == EVENT_LBUTTONDBLCLK) {
        saveImg = true;
    }
}


int main(int argc, char** argv) {
    VideoCapture cap;
    help();
    cv::CommandLineParser parser(argc, argv, "{@input|0|}");
    string input = parser.get<string>("@input");
    if(input.size() == 1 && isdigit(input[0]) )
        cap.open(input[0] - '0');
    else
        cap.open(input);
    if(!cap.isOpened() ) {
        cout << "Cannot initialize capturing by defaulting index 0...\n";
        return 0;
    }
    namedWindow("RRT* Demo", 1);
    setMouseCallback("RRT* Demo", onMouse, 0);
    
    Mat frame, gray;
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    vector<Point2f> points;
    RRTStarPlanner planner;
    vector<RRTStarNode*> path; 
    vector<Point2f> smooth_path;   
    bool planned = false;
    for(;;) {
        cap >> frame;
        if (frame.empty())
            break;
        if (addRemovePt && points.size() < 2) {
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            vector<Point2f> tmp{point};
            cornerSubPix(gray, tmp, Size(31, 31), Size(-1, -1), termcrit);
            points.push_back(tmp[0]);
            addRemovePt = false;
        }
        if (points.size() == 2 && !planned) {
            vector<PolyObstacle> obses;
            planner = RRTStarPlanner(points[0], points[1], obses);
            cout << "Planning...\n";
            planned = planner.Plan(frame);
            path = planner.GetPath();
            
            if (planned) {
                vector<RRTStarNode*> sparse_path;
                for (int i = 0; i < path.size(); i += 5)
                    sparse_path.push_back(path[i]);
                if (path.size() % 5 == 0)
                    sparse_path.push_back(path.back());
                smooth_path = QuadraticBSplineSmoothing(sparse_path);
            }
        }
        if (planned) {
            for (int i = 0; i < int(path.size() - 1); i++)
                line(frame, path[i]->pos, path[i + 1]->pos, Scalar(255,0,0), 2);
            
            for (int i = 0; i < int(smooth_path.size() - 1); i++)
                line(frame, smooth_path[i], smooth_path[i + 1], Scalar(0,0,255), 2);
        }

        for (Point2f pos : points)
            circle(frame, pos, 3, Scalar(0, 255, 0), -1, 8);
        imshow("RRT* Demo", frame);
        char c = (char)waitKey(10);
        if( c == 27 )
            break;
        switch(c) {
        case 'c':
            points.clear();
            break;
        }
    }
    return 0;
}