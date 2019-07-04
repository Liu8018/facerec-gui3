#ifndef FACERECOGNITION_H
#define FACERECOGNITION_H

#include "elm/elm_in_elm_model.h"

class FaceRecognition
{
public:
    FaceRecognition();
    
    //用ELM方法得到前n个候选
    void getCandidatesByELM(const cv::Mat &faceImg, int n, std::vector<std::string> &candidates);
    
    //根据特征相似度(夹角余弦)进行识别
    std::string recognize_byFeat(const cv::Mat &faceImg, const std::vector<std::string> &candidates);
    
private:
    ELM_IN_ELM_Model m_eieModel;
};

#endif // FACERECOGNITION_H
