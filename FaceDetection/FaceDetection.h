#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

/*
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
*/

#include <opencv2/dnn.hpp>

class FaceDetection
{
public:
    FaceDetection();
    
    void detect(const cv::Mat &img, std::vector<cv::Rect> &boxes);
    
private:
    cv::dnn::Net m_net;
};

extern FaceDetection g_faceDT;

#endif // FACEDETECTOR_H
