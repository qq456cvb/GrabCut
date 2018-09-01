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
    float pi[k * 2];
    Vec3f mean[k * 2];
    Mat cov[k * 2];
    Mat img, labels, components;
    
public:
    void init(Mat img, Rect rect);
    void init_gaussian(Rect rect);
};
#endif /* grabcut_hpp */
