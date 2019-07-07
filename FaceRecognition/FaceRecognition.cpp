#include "FaceRecognition.h"
#include "FeatExtraction/FeatExtraction.h"
#include "params.h"
#include "functions.h"

FaceRecognition g_faceRC;

FaceRecognition::FaceRecognition()
{
    
}

std::string FaceRecognition::recognize_resnetOnly(const cv::Mat &faceImg)
{
    //特征提取
    cv::Mat feat;
    g_featEX.resnetEx(faceImg,feat);
    std::vector<std::string> names;
    cv::Mat feats;
    g_featEX.loadFeats_resnet(names,feats);
    
    //相似度：特征夹角余弦
    float maxSim = 0;
    std::string maxSimName;
    for(int i=0;i<feats.rows;i++)
    {
        cv::Mat feat2 = feats.rowRange(i,i+1);
        float a = cv::norm(feat);
        float b = cv::norm(feat2);
        float c = cv::norm(feat-feat2);
        
        float cos = (a*a + b*b - c*c)/(2*a*b);
        
        if(cos > maxSim)
        {
            maxSim = cos;
            maxSimName = names[i];
        }
    }
    
    if(maxSim > MINSIMILARITY_RESNET)
        return maxSimName;
    else
        return "others";
    
    /*
    float minDis = 100;
    std::string minDisName;
    for(int i=0;i<feats.rows;i++)
    {
        cv::Mat feat2 = feats.rowRange(i,i+1);
        float distance = cv::norm(feat-feat2);
        
        if(distance < minDis)
        {
            minDis = distance;
            minDisName = names[i];
        }
    }
    
    std::cout<<"minDistance:"<<minDis<<std::endl;
    return minDisName;
    */
}

void FaceRecognition::getCandidatesByELM(const cv::Mat &faceImg, int n, std::vector<std::string> &candidates)
{
    if(m_eieModel.isEmpty())
        m_eieModel.load(ELM_MODEL_PATH);
    
    //ELM得到候选
    m_eieModel.queryFace(faceImg,n,candidates);
}

std::string FaceRecognition::recognize_byFeat(const cv::Mat &faceImg, 
                                              const std::vector<std::string> &candidates,
                                              std::vector<float> &sims)
{
    //特征提取
    cv::Mat feat;
    g_featEX.extract(faceImg,feat);
    cv::Mat feats;
    std::vector<std::string> names;
    g_featEX.loadFeats(candidates,feats,names);
    
    //相似度：特征夹角余弦
    float maxSim = -1;
    std::string maxSimName;
    std::map<std::string,float> candidate_sim;
    for(size_t i=0;i<candidates.size();i++)
        candidate_sim.insert(std::pair<std::string,float>(candidates[i],0));
    for(int i=0;i<feats.rows;i++)
    {
        cv::Mat feat2 = feats.rowRange(i,i+1);
        float a = cv::norm(feat);
        float b = cv::norm(feat2);
        float c = cv::norm(feat-feat2);
        
        float cos = (a*a + b*b - c*c)/(2*a*b);
        
        if(cos > maxSim)
        {
            maxSim = cos;
            maxSimName = names[i];
        }
        
        float nameMaxSim = candidate_sim.find(names[i])->second;
        if(cos > nameMaxSim)
            candidate_sim.find(names[i])->second = cos;
    }
    
    for(size_t i=0;i<candidates.size();i++)
    {
        float sim = candidate_sim.find(candidates[i])->second;
        sims.push_back(sim);
    }
    
    if(maxSim > MINSIMILARITY_RESNET)
        return maxSimName;
    else
        return "others";
}

void FaceRecognition::EIEtrainNewFace(const cv::Mat &faceImg, std::string name)
{
    m_eieModel.trainNewFace(faceImg,name);
    
    m_eieModel.save();
}
