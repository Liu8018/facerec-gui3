#ifndef FACERECOGNITION_H
#define FACERECOGNITION_H

#include "elm/elm_in_elm_model.h"

class FaceRecognition
{
public:
    FaceRecognition();
    
    //用ELM方法得到前n个候选
    void getCandidatesByELM(const cv::Mat &feat, int n, std::vector<std::string> &candidates, std::vector<float> &sims);
    
    //根据特征相似度(夹角余弦)进行识别
    std::string recognize_byFeat(const cv::Mat &feat, 
                                 const std::vector<std::string> &candidates,
                                 std::vector<float> &sims);
    
    //只使用resnet的特征进行识别
    std::string recognize_resnetOnly(const cv::Mat &faceImg);
    
    //elm训练新图片
    void EIEtrainNewFace(const cv::Mat &faceImg, std::string name);
    
private:
    ELM_IN_ELM_Model m_eieModel;
};

extern FaceRecognition g_faceRC;

#endif // FACERECOGNITION_H
