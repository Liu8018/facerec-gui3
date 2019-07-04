#include "FaceDetection.h"
#include "params.h"

FaceDetection g_faceDT;

FaceDetection::FaceDetection()
{
    m_net = cv::dnn::readNet(
                "./data/face_detection/opencv_face_detector_uint8.pb",
                "./data/face_detection/opencv_face_detector.pbtxt");
}

void FaceDetection::detect(const cv::Mat &img, std::vector<cv::Rect> &boxes)
{
    boxes.clear();
    
    cv::Mat blob = cv::dnn::blobFromImage(img,1.0,cv::Size(300,300),cv::Scalar(104,117,123),true,false);
    
    m_net.setInput(blob);
    cv::Mat detectionMat = m_net.forward();
    cv::Mat detections(detectionMat.size[2], detectionMat.size[3], CV_32F, detectionMat.ptr<float>());
    
    for (int i = 0; i < detections.rows; i++)
    {
        float confidence = detections.at<float>(i, 2);

        if (confidence > FACEDT_CONF_TH)
        {
            int xLeftBottom = static_cast<int>(detections.at<float>(i, 3) * img.cols);
            int yLeftBottom = static_cast<int>(detections.at<float>(i, 4) * img.rows);
            int xRightTop = static_cast<int>(detections.at<float>(i, 5) * img.cols);
            int yRightTop = static_cast<int>(detections.at<float>(i, 6) * img.rows);

            cv::Rect rect(xLeftBottom, yLeftBottom,
                (xRightTop - xLeftBottom),
                (yRightTop - yLeftBottom));
            
            boxes.push_back(rect);
        }
    }
}
