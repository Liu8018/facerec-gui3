#ifndef ELM_MODEL_H
#define ELM_MODEL_H

#include <opencv2/core.hpp>

class ELM_Model
{
public:
    ELM_Model();
    
    //设置隐藏层节点数
    void setHiddenNodes(const int hiddenNodes);
    //设置随机种子
    void setRandomState(int randomState);
    //输入二维数据
    void inputData_2d(std::vector<cv::Mat> &mats, const std::vector<std::vector<bool>> &labels, 
                      const int resizeWidth, const int resizeHeight, const int channels);
    void inputData_2d_test(std::vector<cv::Mat> &mats, const std::vector<std::vector<bool>> &labels);
    //输入一维数据
    void inputData_1d(const cv::Mat &data, const std::vector<std::vector<bool>> &labels);
    void inputData_1d_test(const cv::Mat &data, const std::vector<std::vector<bool>> &labels);
    //设置激活函数
    void setActivation(const std::string method);
    
    //训练
    void fit(int batchSize = -1, bool validating = true, bool verbose = true);
    
    void trainNewImg(const cv::Mat &img, const std::string label);
    void trainNewFace(const cv::Mat &feat, const std::vector<bool> &labelBin);
    
    //查询
    void query(const cv::Mat &mat, std::string &label);
    void query(const cv::Mat &mat, cv::Mat &output);
    void batchQuery(std::vector<cv::Mat> &inputMats, cv::Mat &outputMat);
    void batchQueryFeats(const cv::Mat &feats, cv::Mat &output);
    void queryFeat(const cv::Mat &feat, cv::Mat &output);
    
    //保存和读取模型
    void save(std::string path, std::string K_path="");
    void load(std::string path, std::string K_path="");
    
    void loadStandardDataset(const std::string datasetPath, const float trainSampleRatio,
                             const int resizeWidth, const int resizeHeight, 
                             const int channels, bool shuffle=true);
    
    void loadMnistData(const std::string path, const float trainSampleRatio, bool shuffle=true);
    
    //清除模型参数
    void clear();
    //清除训练数据
    void clearTrainData();
    
    //计算在测试数据上的准确率
    float validate();
    
private:
    int m_randomState;
    
    int m_I;  //输入层节点数
    int m_H;  //隐藏层节点数
    int m_O;  //输出层节点数
    int m_Q;  //输入数据规模
    
    //三层
    cv::Mat m_inputLayerData;   //m_Q×m_I
    cv::Mat m_H_output;         //m_Q×m_H
    cv::Mat m_Target;           //m_Q×m_O
    
    //权重
    cv::Mat m_W_IH;     //m_I×m_H
    cv::Mat m_W_HO;     //m_H×m_O
    //偏置
    cv::Mat m_B_H;      //1×m_H
    
    //在线序列学习中用到的，保留了历史数据的一个矩阵。等于 H的转置*H
    cv::Mat m_K;
    
    std::string m_activationMethod;
    std::string m_defaultActivationMethod;
    
    int m_channels;
    int m_width;
    int m_height;
    
    std::vector<std::string> m_label_string;
    
    int m_Q_test;
    cv::Mat m_Target_test;
    cv::Mat m_inputLayerData_test;
};

#endif // ELM_MODEL_H
