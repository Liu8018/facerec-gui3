#include "FaceRecognition.h"
#include "FeatExtraction/FeatExtraction.h"
#include "params.h"
#include "functions.h"

FaceRecognition::FaceRecognition()
{
    if(access((ELM_MODEL_PATH + "/mainModel.xml").data(),F_OK) == -1)
        handleFaceDb(1);
    m_eieModel.load(ELM_MODEL_PATH);
}

std::string FaceRecognition::recognize_byFeat(const cv::Mat &faceImg, const std::vector<std::string> &candidates)
{
    //特征提取
    cv::Mat feat;
    g_featEX.extract(faceImg,feat);
    
    //相似度：特征夹角余弦
    
}

void FaceRecognition::getCandidatesByELM(const cv::Mat &faceImg, int n, std::vector<std::string> &candidates)
{
    //ELM得到候选
    m_eieModel.queryFace(faceImg,n,candidates);
}
