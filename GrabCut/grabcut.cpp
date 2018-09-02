//
//  grabcut.cpp
//  GrabCut
//
//  Created by Neil on 2018/9/1.
//  Copyright © 2018 Neil. All rights reserved.
//

#include "grabcut.hpp"
#include <iostream>
#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>
#include "graph.h"
#include <limits.h>


void calcGaussians(Mat img, Mat mask, Mat labels, int target, Vec3f &mean, Mat &cov, int &cnt, int label_target=-1) {
    mean = 0;
    cnt = 0;
    float *img_ptr = (float *)img.data;
    uchar *mask_ptr = (uchar *)mask.data;
    uchar *label_ptr = (uchar *)labels.data;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
//            if (label_target >= 0) cout << mask_ptr[i * img.cols + j] << endl;
            if (mask_ptr[i * img.cols + j] == target && (label_target < 0 || label_ptr[i * img.cols + j] == label_target)) {
                mean += 1. / (++cnt) * (Vec3f(&img_ptr[(i * img.cols + j) * 3]) - mean);
            }
        }
    }
    // get fg and bg cov
    cov = Mat(3, 3, CV_32F, Scalar(0));
    cnt = 0;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (mask_ptr[i * img.cols + j] == target && (label_target < 0 || label_ptr[i * img.cols + j] == label_target)) {
                Mat tmp = Mat(Vec3f(&img_ptr[(i * img.cols + j) * 3]) - mean);
                cov += 1. / (++cnt) * (tmp * tmp.t() - cov);
            }
        }
    }
}


void GrabCut::init(cv::Mat img, cv::Rect rect) {
    this->rect = rect;
    img.convertTo(this->img, CV_32FC3, 1. / 255);
    this->labels = Mat(img.rows, img.cols, CV_8U, Scalar(0));
    this->bg_components = Mat(img.rows, img.cols, CV_8U, Scalar(0));
    this->fg_components = Mat(img.rows, img.cols, CV_8U, Scalar(0));
    
    for (int i = rect.y;  i < rect.y + rect.height; i++) {
        for (int j = rect.x; j < rect.x + rect.width; j++) {
            labels.at<unsigned char>(i, j) = 1;
        }
    }
    this->init_gaussian(rect);
    int iter = 0;
    while (iter++ < 100) {
        assign_GMM_components();
        update_GMM_params();
        estimate_segmentation();
    }
//    imshow("test", labels);
}

void GrabCut::init_gaussian(Rect rect) {
    // get fg and bg mean
    Vec3f fg_mean, bg_mean;
    Mat fg_cov, bg_cov;
    int fg_cnt, bg_cnt;
    calcGaussians(img, labels, Mat(), 0, bg_mean, bg_cov, bg_cnt);
    calcGaussians(img, labels, Mat(), 1, fg_mean, fg_cov, fg_cnt);
//    Vec3f fg_mean = 0, bg_mean = 0;
//    int fg_cnt = 0, bg_cnt = 0;
//    float *img_ptr = (float *)this->img.data;
//    uchar *label_ptr = (uchar *)labels.data;
//    for (int i = 0; i < img.rows; i++) {
//        for (int j = 0; j < img.cols; j++) {
//            if (label_ptr[i * img.cols + j] == 1) {
//                fg_mean += 1. / (++fg_cnt) * (Vec3f(&img_ptr[(i * img.cols + j) * 3]) - fg_mean);
//            } else {
//                bg_mean += 1. / (++bg_cnt) * (Vec3f(&img_ptr[(i * img.cols + j) * 3]) - bg_mean);
//            }
//        }
//    }
//    // get fg and bg cov
//    Mat fg_cov(3, 3, CV_32F, Scalar(0)), bg_cov(3, 3, CV_32F, Scalar(0));
//    fg_cnt = 0;
//    bg_cnt = 0;
//    for (int i = 0; i < img.rows; i++) {
//        for (int j = 0; j < img.cols; j++) {
//            if (label_ptr[i * img.cols + j] == 1) {
//                Mat tmp = Mat(Vec3f(&img_ptr[(i * img.cols + j) * 3]) - fg_mean);
//                fg_cov += 1. / (++fg_cnt) * (tmp * tmp.t() - fg_cov);
//            } else {
//                Mat tmp = Mat(Vec3f(&img_ptr[(i * img.cols + j) * 3]) - bg_mean);
//                bg_cov += 1. / (++bg_cnt) * (tmp * tmp.t() - bg_cov);
//            }
//        }
//    }
    cout << fg_mean << endl;
    cout << bg_mean << endl;
    cout << fg_cov << endl;
    cout << bg_cov << endl;
    for (int i = 0; i < k; i++) {
        int x = rand() % rect.width + rect.x;
        int y = rand() % rect.height + rect.y;
        mean[1][i] = img.at<Vec3f>(y, x);
        cov[1][i] = fg_cov;
    }
    int i = 0;
    while (i < k) {
        int x = rand() % img.cols;
        int y = rand() % img.rows;
        if (x >= rect.x && x < rect.x + rect.width && y >= rect.y && y < rect.y + rect.height) {
            continue;
        }
        mean[0][i] = img.at<Vec3f>(y, x);
        cov[0][i] = bg_cov;
        i++;
    }
    
    for (int i = 0; i < k; i++) {
        pi[0][i] = 1.f / k;
        pi[1][i] = 1.f / k;
    }
}

void GrabCut::assign_GMM_components() {
    float *img_ptr = (float *)this->img.data;
    uchar *label_ptr = (uchar *)labels.data;
    uchar *bg_comp_ptr = (uchar *)bg_components.data;
    uchar *fg_comp_ptr = (uchar *)fg_components.data;
    for (int i = 0;  i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
//            int label = label_ptr[i * img.cols + j];
            Mat pixel = Mat(Vec3f(&img_ptr[3 * (i * img.cols + j)]));
            for (int l = 0; l < 2; l++) {
                float min_d = numeric_limits<float>::max();
                int min_idx = -1;
                for (int n = 0; n < k; n++) {
                    Mat tmp = pixel - mean[l][n];
                    tmp = (tmp.t() * cov[l][n].inv() * tmp);
                    //                cout << tmp.rows << ", " << tmp.cols << endl;
                    float d = -log(pi[l][n] + eps) + 0.5 * log(determinant(cov[l][n]) + eps) + 0.5 * *(float*)tmp.data;
                    if (d < min_d) {
                        min_d = d;
                        min_idx = n;
                    }
                }
                if (l == 0) {
                    bg_comp_ptr[i * img.cols + j] = min_idx;
                } else {
                    fg_comp_ptr[i * img.cols + j] = min_idx;
                }
            }
        }
    }
    
    Mat tmp;
    bg_components.convertTo(tmp, CV_8U, 51);
    imshow("bg_comp", tmp);
    fg_components.convertTo(tmp, CV_8U, 51);
    imshow("fg_comp", tmp);
    waitKey(30);
}

void GrabCut::update_GMM_params() {
    vector<int> bg_cnts, fg_cnts;
    Mat tmp;
    labels.convertTo(tmp, CV_8U, 255);
    imshow("labels", tmp);
    waitKey(30);
    for (int i = 0; i < k; i++) {
        int cnt;
        calcGaussians(img, bg_components, labels, i, mean[0][i], cov[0][i], cnt, 0);
        bg_cnts.push_back(cnt);
    }
    for (int i = 0; i < k; i++) {
        int cnt;
        calcGaussians(img, fg_components, labels, i, mean[1][i], cov[1][i], cnt, 1);
        fg_cnts.push_back(cnt);
    }
    
    int bg_total_cnts = accumulate(bg_cnts.begin(), bg_cnts.end(), 0);
    int fg_total_cnts = accumulate(fg_cnts.begin(), fg_cnts.end(), 0);
    cout << fg_total_cnts << endl;
    for (int i = 0; i < k; i++) {
        pi[0][i] = bg_cnts[i] / float(bg_total_cnts);
    }
    for (int i = 0; i < k; i++) {
        pi[1][i] = fg_cnts[i] / float(fg_total_cnts);
    }
}

void GrabCut::estimate_segmentation() {
    uchar *label_ptr = (uchar *)labels.data;
    uchar *bg_comp_ptr = (uchar *)bg_components.data;
    uchar *fg_comp_ptr = (uchar *)fg_components.data;
    float *img_ptr = (float *)img.data;
    typedef Graph<double,double,double> GraphType;
    GraphType *g = new GraphType(/*estimated # of nodes*/ img.rows * img.cols, /*estimated # of edges*/ img.rows * img.cols * 8 / 2);
    
    for (int i = 0; i < img.rows * img.cols; i++) {
        g -> add_node();
    }
    
    // eight weight, get beta first
    float beta = 0.f;
    int cnt = 0;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            int idx = i * img.cols + j;
            for (int m = max(0, i - 1); m <= min(img.rows - 1, i); m++) {
                for (int n = max(0, j - 1); n <= min(img.cols - 1, j + 1); n++) {
                    if (m == i && n >= j) {
                        continue;
                    }
                    int nb_idx = m * img.cols + n;
                    beta += 1. / (++cnt) * (Vec3f(&img_ptr[idx * 3]).dot(Vec3f(&img_ptr[nb_idx * 3])) - beta);
                }
            }
        }
    }
    cout << beta << endl;
    float max_w = 0;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            int idx = i * img.cols + j;
            float sum_w = 0;
            for (int m = max(0, i - 1); m <= min(img.rows - 1, i + 1); m++) {
                for (int n = max(0, j - 1); n <= min(img.cols - 1, j + 1); n++) {
                    if (m == i && n == j) {
                        continue;
                    }
                    int nb_idx = m * img.cols + n;
                    float d = Vec3f(&img_ptr[idx * 3]).dot(Vec3f(&img_ptr[nb_idx * 3]));
                    float w = gma * exp(-beta * d);
//                    cout << w << endl;
                    sum_w += w;
                    g->add_edge(idx, nb_idx, w, 0);
                }
            }
            max_w = max(sum_w, max_w);
            if (i >= rect.y && i < rect.y + rect.height && j >= rect.x && j < rect.x + rect.width) {
                
                Mat tmp(Vec3f(&img_ptr[idx * 3]) - mean[0][bg_comp_ptr[idx]]);
                tmp = (tmp.t() * cov[0][bg_comp_ptr[idx]].inv() * tmp);
                float d_bg = -log(pi[0][bg_comp_ptr[idx]] + eps) + 0.5 * log(determinant(cov[0][bg_comp_ptr[idx]]) + eps) + 0.5 * *(float*)tmp.data;
                
                tmp = Mat(Vec3f(&img_ptr[idx * 3]) - mean[1][fg_comp_ptr[idx]]);
                tmp = (tmp.t() * cov[1][fg_comp_ptr[idx]].inv() * tmp);
                float d_fg = -log(pi[1][fg_comp_ptr[idx]] + eps) + 0.5 * log(determinant(cov[1][fg_comp_ptr[idx]]) + eps) + 0.5 * *(float*)tmp.data;
//                cout << determinant(cov[0][bg_comp_ptr[idx]]) << ", " << determinant(cov[1][fg_comp_ptr[idx]]) << endl;
                g->add_tweights(idx, d_bg, d_fg);
            }
        }
    }
    
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            int idx = i * img.cols + j;
            if (!(i >= rect.y && i < rect.y + rect.height && j >= rect.x && j < rect.x + rect.width)) {
                g -> add_tweights(idx, 0.f, max_w + 1.f);
            }
        }
    }
    
    float flow = g -> maxflow();
    
    printf("Flow = %f\n", flow);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            int idx = i * img.cols + j;
            if (g->what_segment(idx) == GraphType::SOURCE) {
                label_ptr[idx] = 1;
            } else {
                label_ptr[idx] = 0;
            }
        }
    }
    Mat tmp;
//    labels.convertTo(tmp, CV_8U, 255);
//    imshow("labels", tmp);
//    waitKey(30);
//    printf("Minimum cut:\n");
//    if (g->what_segment(0) == GraphType::SOURCE)
//        printf("node0 is in the SOURCE set\n");
//    else
//        printf("node0 is in the SINK set\n");
//    if (g->what_segment(1) == GraphType::SOURCE)
//        printf("node1 is in the SOURCE set\n");
//    else
//        printf("node1 is in the SINK set\n");
    
    delete g;
}
