//
//  grabcut.cpp
//  GrabCut
//
//  Created by Neil on 2018/9/1.
//  Copyright © 2018 Neil. All rights reserved.
//

#include "grabcut.hpp"
#include <iostream>

void GrabCut::init(cv::Mat img, cv::Rect rect) {
    img.convertTo(this->img, CV_32FC3, 1. / 255);
    this->labels = Mat(img.rows, img.cols, CV_8U, Scalar(0));
    this->components = Mat(img.rows, img.cols, CV_8U, Scalar(0));
    
    for (int i = rect.y;  i < rect.y + rect.height; i++) {
        for (int j = rect.x; j < rect.x + rect.width; j++) {
            labels.at<unsigned char>(i, j) = 1;
        }
    }
    this->init_gaussian(rect);
//    imshow("test", labels);
}

void GrabCut::init_gaussian(Rect rect) {
    // get fg and bg mean
    Vec3f fg_mean = 0, bg_mean = 0;
    int fg_cnt = 0, bg_cnt = 0;
    float *img_ptr = (float *)this->img.data;
    uchar *label_ptr = (uchar *)labels.data;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (label_ptr[i * img.cols + j] == 1) {
                fg_mean += 1. / (++fg_cnt) * (Vec3f(&img_ptr[(i * img.cols + j) * 3]) - fg_mean);
            } else {
                bg_mean += 1. / (++bg_cnt) * (Vec3f(&img_ptr[(i * img.cols + j) * 3]) - bg_mean);
            }
        }
    }
    // get fg and bg cov
    Mat fg_cov(3, 3, CV_32F, Scalar(0)), bg_cov(3, 3, CV_32F, Scalar(0));
    fg_cnt = 0;
    bg_cnt = 0;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (label_ptr[i * img.cols + j] == 1) {
                Mat tmp = Mat(Vec3f(&img_ptr[(i * img.cols + j) * 3]) - fg_mean);
                fg_cov += 1. / (++fg_cnt) * (tmp * tmp.t() - fg_cov);
            } else {
                Mat tmp = Mat(Vec3f(&img_ptr[(i * img.cols + j) * 3]) - bg_mean);
                bg_cov += 1. / (++bg_cnt) * (tmp * tmp.t() - bg_cov);
            }
        }
    }
    cout << fg_mean << endl;
    cout << bg_mean << endl;
    cout << fg_cov << endl;
    cout << bg_cov << endl;
    for (int i = 0; i < k; i++) {
        int x = rand() % rect.width + rect.x;
        int y = rand() % rect.height + rect.y;
        mean[k + i] = img.at<Vec3f>(y, x);
        cov[k + i] = fg_cov;
    }
    for (int i = 0; i < k; i++) {
        int x = rand() % img.cols;
        int y = rand() % img.rows;
        if (x >= rect.x && x < rect.x + rect.width && y >= rect.y && y < rect.y + rect.height) {
            continue;
        }
        mean[i] = img.at<Vec3f>(y, x);
        cov[i] = bg_cov;
    }
    
    for (int i = 0; i < k * 2; i++) {
        pi[i] = 1.f / k;
    }
    
}
