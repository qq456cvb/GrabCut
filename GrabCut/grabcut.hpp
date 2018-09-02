//
//  grabcut.hpp
//  GrabCut
//
//  Created by Neil on 2018/9/1.
//  Copyright © 2018 Neil. All rights reserved.
//

#ifndef grabcut_hpp
#define grabcut_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

const int k = 5;

class GrabCut {
    float gma = 50.f;
    float pi[2][k];
    float eps = 1e-7f;
    Vec3f mean[2][k];
    Mat cov[2][k];
    Mat img, labels, fg_components, bg_components;
    Rect rect;
    
public:
    void init(Mat img, Rect rect);
    void init_gaussian(Rect rect);
    void assign_GMM_components();
    void update_GMM_params();
    void estimate_segmentation();
};
#endif /* grabcut_hpp */
