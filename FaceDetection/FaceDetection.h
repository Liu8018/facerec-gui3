#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

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
