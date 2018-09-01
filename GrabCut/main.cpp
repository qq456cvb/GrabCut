//
//  main.cpp
//  GrabCut
//
//  Created by Neil on 2018/9/1.
//  Copyright © 2018 Neil. All rights reserved.
//

#include <iostream>

/** This section shows how to use the library to compute
a minimum cut on the following graph:

        SOURCE
        /             \
    1/                \2
    /       3           \
node0 -----> node1
    |   <-----      |
    |      4         |
    \                /
    5\            /6
        \        /
        SINK

**/

#include <stdio.h>
#include "graph.h"
#include "grabcut.hpp"
#include <opencv2/opencv.hpp>
using namespace cv;

void test() {
    typedef Graph<int,int,int> GraphType;
    GraphType *g = new GraphType(/*estimated # of nodes*/ 2, /*estimated # of edges*/ 1);
    
    g -> add_node();
    g -> add_node();
    
    g -> add_tweights( 0,   /* capacities */  1, 5 );
    g -> add_tweights( 1,   /* capacities */  2, 6 );
    g -> add_edge( 0, 1,    /* capacities */  3, 4 );
    
    int flow = g -> maxflow();
    
    printf("Flow = %d\n", flow);
    printf("Minimum cut:\n");
    if (g->what_segment(0) == GraphType::SOURCE)
        printf("node0 is in the SOURCE set\n");
    else
        printf("node0 is in the SINK set\n");
    if (g->what_segment(1) == GraphType::SOURCE)
        printf("node1 is in the SOURCE set\n");
    else
        printf("node1 is in the SINK set\n");
    
    delete g;
}

void mouseHandler(int event, int x, int y, int flags, void* param)
{
    Mat raw = *(Mat *)param;
    static bool dragging = false;
    static Point start = Point(0, 0);
    static Point end = Point(0, 0);
    if (event == CV_EVENT_LBUTTONDOWN && !dragging)
    {
        /* left button clicked. ROI selection begins */
        start = Point(x, y);
        dragging = true;
    }
    
    if (event == CV_EVENT_MOUSEMOVE && dragging)
    {
        /* mouse dragged. ROI being selected */
        cv::Mat tmp = raw.clone();
        end = Point(x, y);
        cv::rectangle(tmp, start, end, CV_RGB(255, 0, 0), 3, 8, 0);
        cv::imshow("img", tmp);
    }
    
    if (event == CV_EVENT_LBUTTONUP && dragging)
    {
        GrabCut gc;
        gc.init(raw, Rect(start.x, start.y, end.x - start.x, end.y - start.y));
        dragging = false;
        
    }
}


int main()
{
    auto img = imread("./data_GT/banana1.bmp");
    imshow("img", img);
    cv::setMouseCallback("img",mouseHandler,&img);
    auto key = waitKey();
    
    return 0;
}

