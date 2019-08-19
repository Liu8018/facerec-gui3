#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

//用yushiqi的人脸检测库

#include "facedetectcnn.h"
#include <opencv2/core.hpp>

class FaceDetection
{
public:
    FaceDetection();
    
    void detect(const cv::Mat &img, std::vector<cv::Rect> &boxes);
    
private:
    int * pResults;
    unsigned char * pBuffer;
    
    int resizeWidth;
    float resizeRatio;
};

//用opencv的dnn模块
/*
#include <opencv2/dnn.hpp>

class FaceDetection
{
public:
    FaceDetection();
    
    void detect(const cv::Mat &img, std::vector<cv::Rect> &boxes);
    
private:
    cv::dnn::Net m_net;
};
*/

extern FaceDetection g_faceDT;

#endif // FACEDETECTOR_H
