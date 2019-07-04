#include "elm_model.h"
#include <iostream>
#include <ctime>
#include "ELM_functions.h"
#include <opencv2/imgproc.hpp>

ELM_Model::ELM_Model()
{
    m_I = -1;
    m_H = -1;
    m_O = -1;
    m_Q = -1;
    
    m_defaultActivationMethod = "sigmoid";
    m_randomState = -1;
}

void ELM_Model::clear()
{
    m_W_IH.release();
    m_B_H.release();
    m_W_HO.release();
    m_K.release();
}

void ELM_Model::inputData_2d(std::vector<cv::Mat> &mats, const std::vector<std::vector<bool>> &labels, 
                             const int resizeWidth, const int resizeHeight, const int channels)
{
    if(mats.empty())
        return;
    
    m_channels = channels;
    m_width = resizeWidth;
    m_height = resizeHeight;
    
    //确定输入数据规模
    m_Q = mats.size();
    //确定输出层节点数
    m_O = labels[0].size();
    
    //转化label为target
    label2target(labels,m_Target);
    //m_inputLayerData.create(cv::Size(m_I,m_Q),CV_32F);
    for(int i=0;i<mats.size();i++)
        cv::resize(mats[i],mats[i],cv::Size(m_width,m_height));
    mats2lines(mats,m_inputLayerData,m_channels);
    
    normalize(m_inputLayerData);
    
    //确定输入层节点数
    m_I = m_inputLayerData.cols;
    
//std::cout<<"m_Target:\n"<<m_Target<<std::endl;
//std::cout<<"m_inputLayerData:\n"<<m_inputLayerData<<std::endl;

}

void ELM_Model::inputData_1d(const cv::Mat &data, const std::vector<std::vector<bool> > &labels)
{
    if(data.empty())
        return;
    
    m_channels = 1;
    
    m_Q = data.rows;
    m_O = labels[0].size();
    
    label2target(labels,m_Target);
    
    data.copyTo(m_inputLayerData);
    
    normalize(m_inputLayerData);
    
    m_I = m_inputLayerData.cols;
}

void ELM_Model::inputData_2d_test(std::vector<cv::Mat> &mats, const std::vector<std::vector<bool> > &labels)
{
    if(mats.empty())
        return;
    
    m_Q_test = mats.size();
    
    label2target(labels,m_Target_test);
    
    //m_inputLayerData_test.create(cv::Size(m_I,m_Q_test),CV_32F);
    for(int i=0;i<mats.size();i++)
        cv::resize(mats[i],mats[i],cv::Size(m_width,m_height));
    mats2lines(mats,m_inputLayerData_test,m_channels);
    
    normalize(m_inputLayerData_test);
}

void ELM_Model::inputData_1d_test(const cv::Mat &data, const std::vector<std::vector<bool> > &labels)
{
    if(data.empty())
        return;
    
    m_Q_test = data.rows;
    
    label2target(labels,m_Target_test);
    
    data.copyTo(m_inputLayerData_test);
    
    normalize(m_inputLayerData_test);
}

void ELM_Model::loadMnistData(const std::string path, const float trainSampleRatio, bool shuffle)
{
    std::vector<cv::Mat> trainImgs;
    std::vector<cv::Mat> testImgs;
    
    std::vector<std::vector<bool>> trainLabelBins;
    std::vector<std::vector<bool>> testLabelBins;
    loadMnistData_csv(path,trainSampleRatio,trainImgs,testImgs,trainLabelBins,testLabelBins,shuffle);
    
    inputData_2d(trainImgs,trainLabelBins,28,28,1);
    inputData_2d_test(testImgs,testLabelBins);
}

void ELM_Model::setHiddenNodes(const int hiddenNodes)
{
    m_H = hiddenNodes;
}

void ELM_Model::setActivation(const std::string method)
{
    m_activationMethod = method;
}

void ELM_Model::setRandomState(int randomState)
{
    m_randomState = randomState;
}

void ELM_Model::fit(int batchSize, bool validating, bool verbose)
{
    std::cout<<"【elm训练开始】--------------------------------------"<<std::endl;
    
    if(m_inputLayerData.empty())
        return;
    
    //检查隐藏层节点数是否被设置
    if(m_H == -1)
        m_H = m_Q/2;
    
    //检查是否设置batchSize
    if(batchSize == -1)
        batchSize = m_Q;
    
    //输出信息
    if(verbose)
    {
        std::cout<<"Q:"<<m_Q<<std::endl;
        std::cout<<"batchSize:"<<batchSize<<std::endl;
        std::cout<<"I:"<<m_I<<std::endl;
        std::cout<<"H:"<<m_H<<std::endl;
        std::cout<<"O:"<<m_O<<std::endl;
    }
    
    //初次训练的初始化
    if(m_W_IH.empty())
    {
        //分配空间
        m_W_IH.create(cv::Size(m_H,m_I),CV_32F);
        m_W_HO.create(cv::Size(m_O,m_H),CV_32F);
        m_B_H.create(cv::Size(m_H,1),CV_32F);
        
        //K初始化
        m_K.create(cv::Size(m_H,m_H),CV_32F);
        m_K = cv::Scalar(0);
        
        //输出权重初始化
        m_W_HO = cv::Scalar(0);
        
        //随机产生IH权重和H偏置
        int randomState;
        if(m_randomState != -1)
            randomState = m_randomState;
        else
            randomState = (unsigned)time(nullptr);
        randomGenerate(m_W_IH,m_W_IH.size(),randomState);
        randomGenerate(m_B_H,m_B_H.size(),randomState+1);
    }
    
    int trainedRatio=0;
    for(int i=0;i+batchSize<=m_Q;i+=batchSize)
    {
        //取批次训练部分数据
        cv::Mat inputBatchROI = m_inputLayerData(cv::Range(i,i+batchSize),cv::Range(0,m_I));
        cv::Mat targetBatchROI = m_Target(cv::Range(i,i+batchSize),cv::Range(0,m_O));
        
        //计算H输出
            //输入乘权重
        m_H_output = inputBatchROI * m_W_IH;
            //加上偏置
        addBias(m_H_output,m_B_H);
            //激活
        if(m_activationMethod.empty())
            m_activationMethod = m_defaultActivationMethod;
        activate(m_H_output,m_activationMethod);
            //迭代更新K
        m_K = m_K + m_H_output.t() * m_H_output;
        
        //迭代更新HO权重
        m_W_HO = m_W_HO + m_K.inv(1) * m_H_output.t() * (targetBatchROI - m_H_output*m_W_HO);
        
        //输出信息
        if(verbose)
        {
            int ratio = (i+batchSize)/(float)m_Q*100;
            if( ratio - trainedRatio >= 1)
            {
                trainedRatio = ratio;
                
                //计算并输出在该批次训练数据上的准确率
                cv::Mat output = m_H_output * m_W_HO;
                float score = calcScore(output,targetBatchROI);
                std::cout<<"Score on batch training data:"<<score<<std::endl;
                
                //计算在测试数据上的准确率
                if(validating)
                    validate();
            }
        }
    }
    
    std::cout<<"【elm训练结束】--------------------------------------"<<std::endl;
    
//std::cout<<"m_W_IH:\n"<<m_W_IH<<std::endl;
//std::cout<<"m_B_H:\n"<<m_B_H<<std::endl;
//std::cout<<"m_H_output:\n"<<m_H_output<<std::endl;
//std::cout<<"m_W_HO:\n"<<m_W_HO<<std::endl;
//std::cout<<"test:\n"<<m_H_output * m_W_HO<<"\n"<<m_Target<<std::endl;
}

float ELM_Model::validate()
{
    //计算在测试数据上的准确率
    if(!m_inputLayerData_test.empty())
    {
        std::cout<<"validate:------------------"<<std::endl;
                
        cv::Mat m1 = m_inputLayerData_test * m_W_IH;
        addBias(m1,m_B_H);
        if(m_activationMethod.empty())
            m_activationMethod = m_defaultActivationMethod;
        activate(m1,m_activationMethod);
        cv::Mat output = m1 * m_W_HO;
        float finalScore_test = calcScore(output,m_Target_test);
        
        std::cout<<"Score on validation data:"<<finalScore_test<<std::endl;
        return finalScore_test;
    }
    else
        return 0;
}

void ELM_Model::query(const cv::Mat &mat, std::string &label)
{
    //转化为一维数据
    cv::Mat inputLine(cv::Size(m_width*m_channels*m_height,1),CV_32F);
    cv::Mat tmpImg;
    cv::resize(mat,tmpImg,cv::Size(m_width,m_height));
    mat2line(tmpImg,inputLine,m_channels);
    normalize_img(inputLine);
    
    //乘权重，加偏置，激活
    cv::Mat H = inputLine * m_W_IH;
    addBias(H,m_B_H);
    activate(H,m_activationMethod);
    
    //计算输出
    cv::Mat output = H * m_W_HO;
    
    int id = getMaxId(output);
    label = m_label_string[id];
}

void ELM_Model::query(const cv::Mat &mat, cv::Mat &output)
{
    //转化为一维数据
    cv::Mat inputLine;//(cv::Size(m_width*m_channels*m_height,1),CV_32F);
    cv::Mat tmpImg;
    cv::resize(mat,tmpImg,cv::Size(m_width,m_height));
    
    //debug
    //cv::imshow("input img",tmpImg);
    
    mat2line(tmpImg,inputLine,m_channels);
    
    //debug
    //std::cout<<"inputLine:\n"<<inputLine<<std::endl;
    
    normalize(inputLine);
    
    //debug
    //std::cout<<"nomalized inputLine:\n"<<inputLine<<std::endl;
    
    //乘权重，加偏置，激活
    cv::Mat H = inputLine * m_W_IH;
    addBias(H,m_B_H);
    activate(H,m_activationMethod);
    
    //计算输出
    output = H * m_W_HO;
    
    //debug
    //std::cout<<"output:\n"<<output<<std::endl;
    
    //cv::waitKey();
}

void ELM_Model::batchQuery(std::vector<cv::Mat> &inputMats, cv::Mat &outputMat)
{
    for(int i=0;i<inputMats.size();i++)
        cv::resize(inputMats[i],inputMats[i],cv::Size(m_width,m_height));
    
    cv::Mat inputLayerData(cv::Size(m_width*m_height*m_channels,inputMats.size()),CV_32F);
    normalize(inputLayerData);

    cv::Mat H = inputLayerData * m_W_IH;
    addBias(H,m_B_H);
    activate(H,m_activationMethod);

    outputMat = H * m_W_HO;
}

void ELM_Model::save(std::string path, std::string K_path)
{
    if(m_W_HO.empty())
        return;
    
    cv::FileStorage fswrite(path,cv::FileStorage::WRITE);
    
    fswrite<<"channels"<<m_channels;
    fswrite<<"width"<<m_width;
    fswrite<<"height"<<m_height;
    fswrite<<"m_H"<<m_H;
    fswrite<<"W_IH"<<m_W_IH;
    fswrite<<"W_HO"<<m_W_HO;
    fswrite<<"B_H"<<m_B_H;
    fswrite<<"activationMethod"<<m_activationMethod;
    fswrite<<"label_string"<<m_label_string;
    
    if(K_path != "")
    {
        cv::FileStorage K_fswrite(K_path,cv::FileStorage::WRITE);
        K_fswrite<<"K"<<m_K;
        K_fswrite.release();
    }
    
    fswrite.release();
}

void ELM_Model::load(std::string path, std::string K_path)
{
    cv::FileStorage fsread(path,cv::FileStorage::READ);
    
    fsread["channels"]>>m_channels;
    fsread["width"]>>m_width;
    fsread["height"]>>m_height;
    fsread["m_H"]>>m_H;
    fsread["W_IH"]>>m_W_IH;
    fsread["W_HO"]>>m_W_HO;
    fsread["B_H"]>>m_B_H;
    fsread["activationMethod"]>>m_activationMethod;
    fsread["label_string"]>>m_label_string;
    
    if(K_path != "")
    {
        cv::FileStorage K_fsread(K_path,cv::FileStorage::READ);
        K_fsread["K"]>>m_K;
        K_fsread.release();
    }
    
    fsread.release();
}

void ELM_Model::loadStandardDataset(const std::string datasetPath, const float trainSampleRatio,
                                    const int resizeWidth, const int resizeHeight, 
                                    const int channels, bool shuffle)
{
    m_channels = channels;
    
    std::vector<cv::Mat> trainImgs;
    std::vector<cv::Mat> testImgs;
    
    std::vector<std::vector<bool>> trainLabelBins;
    std::vector<std::vector<bool>> testLabelBins;
    
    inputImgsFrom(datasetPath,m_label_string,trainImgs,testImgs,trainLabelBins,testLabelBins,trainSampleRatio,channels,shuffle);

    inputData_2d(trainImgs,trainLabelBins,resizeWidth,resizeHeight,channels);
    inputData_2d_test(testImgs,testLabelBins);
}

void ELM_Model::clearTrainData()
{
    if(m_inputLayerData.empty())
        return;
    
    if(!m_inputLayerData.empty())
        m_inputLayerData.release();
    if(!m_H_output.empty())
        m_H_output.release();
    if(!m_Target.empty())
        m_Target.release();
    if(!m_inputLayerData_test.empty())
        m_inputLayerData_test.release();
    if(!m_Target_test.empty())
        m_Target_test.release();
}

void ELM_Model::trainNewImg(const cv::Mat &img, const std::string label)
{
    clearTrainData();
    
    std::vector<bool> labelBin(m_O,0);
    for(int i=0;i<m_label_string.size();i++)
        if(label == m_label_string[i])
        {
            labelBin[i] = 1;
            break;
        }
    std::vector<std::vector<bool>> trainLabels;
    trainLabels.push_back(labelBin);
    
    std::vector<cv::Mat> trainMats;
    trainMats.push_back(img);
    
    inputData_2d(trainMats,trainLabels,m_width,m_height,m_channels);
    
    fit();
}
