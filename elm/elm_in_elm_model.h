#ifndef ELM_IN_ELM_MODEL_H
#define ELM_IN_ELM_MODEL_H

#include "elm_model.h"

class ELM_IN_ELM_Model
{
public:
    ELM_IN_ELM_Model();
    ELM_IN_ELM_Model(const int n_models,const std::string modelDir);
    
    void setInitPara(const int n_models,const std::string modelDir);
    
    void setSubModelHiddenNodes(const int modelId, const int n_nodes);
    
    void loadStandardDataset(const std::string path, const float trainSampleRatio,
                             const int resizeWidth, const int resizeHeight, 
                             const int channels, bool shuffle=true);
    
    void loadStandardFaceDataset(const std::string path, const float trainSampleRatio,
                                 const int resizeWidth, const int resizeHeight, bool shuffle=true);
    
    void loadMnistData(const std::string path, const float trainSampleRatio, bool shuffle=true);
    
    void fitSubModels(int batchSize = -1, bool validating = true, bool verbose = true);
    void fitSubModels_faceFeat(int batchSize = -1, bool validating = true, bool verbose = true);
    void fitMainModel(int batchSize = -1, bool validating = true, bool verbose = true);
    void fitMainModel_faceFeat(int batchSize = -1, bool validating = false, bool verbose = true);
    
    void trainNewImg(const cv::Mat &img, const std::string label);
    void trainNewFace(const cv::Mat &img, const std::string label);
    
    void save();
    void load(std::string modelDir);
    
    void query(const cv::Mat &mat, std::string &label);
    void queryFace(const cv::Mat &mat, std::string &label);
    
    //得到前n个最大值ID
    void query(const cv::Mat &mat, int n, std::vector<std::string> &names);
    void queryFace(const cv::Mat &feat, int n, std::vector<std::string> &names);
    void queryFace(const cv::Mat &feat, int n, std::vector<std::string> &names, std::vector<float> &sims);
    
    void clearTrainData();
    void clearTrainData_feat();
    
    //计算在测试数据上的准确率
    float validate();
    
    bool isEmpty();
    
private:
    std::vector<std::vector<cv::Mat>> m_allFeats;
    
    int m_n_models;
    std::vector<int> m_subModelHiddenNodes;
    
    ELM_Model m_subModelToTrain;
    
    std::vector<ELM_Model> m_subModels;
    
    std::string m_modelPath;
    int m_width;
    int m_height;
    int m_channels;
    int m_Q;
    int m_C;
    
    cv::Mat m_K;
    
    cv::Mat m_F;
    
    std::vector<std::string> m_label_string;
    
    std::vector<cv::Mat> m_trainImgs;
    std::vector<cv::Mat> m_testImgs;
    std::vector<std::vector<bool>> m_trainLabelBins;
    std::vector<std::vector<bool>> m_testLabelBins;
    
    cv::Mat m_faceFeats;
    cv::Mat m_faceFeats_test;
};

#endif // ELM_IN_ELM_MODEL_H
