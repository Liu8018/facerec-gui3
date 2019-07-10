#include "elm_in_elm_model.h"
#include "functions.h"
#include "ELM_functions.h"
#include <iostream>

ELM_IN_ELM_Model::ELM_IN_ELM_Model()
{
    
}
ELM_IN_ELM_Model::ELM_IN_ELM_Model(const int n_models, const std::string modelDir)
{
    m_n_models = n_models;
    m_subModelHiddenNodes.resize(m_n_models);
    
    for(int i=0;i<m_n_models;i++)
        m_subModelHiddenNodes[i] = -1;
    
    m_modelPath = modelDir;
    if(m_modelPath[m_modelPath.length()-1] != '/')
        m_modelPath.append("/");
}

void ELM_IN_ELM_Model::setInitPara(const int n_models, const std::string modelDir)
{
    m_n_models = n_models;
    m_subModelHiddenNodes.resize(m_n_models);
    
    for(int i=0;i<m_n_models;i++)
        m_subModelHiddenNodes[i] = -1;
    
    m_modelPath = modelDir;
    if(m_modelPath[m_modelPath.length()-1] != '/')
        m_modelPath.append("/");
}

void ELM_IN_ELM_Model::setSubModelHiddenNodes(const int modelId, const int n_nodes)
{
    if(modelId == -1)
    {
        for(int i=0;i<m_n_models;i++)
            m_subModelHiddenNodes[i] = n_nodes;
    }
    else
        m_subModelHiddenNodes[modelId] = n_nodes;
}

void ELM_IN_ELM_Model::loadStandardDataset(const std::string path, const float trainSampleRatio, 
                                           const int resizeWidth, const int resizeHeight, const int channels, bool shuffle)
{
    m_width = resizeWidth;
    m_height = resizeHeight;
    m_channels = channels;
    
    inputImgsFrom(path,m_label_string,m_trainImgs,
                  m_testImgs,m_trainLabelBins,m_testLabelBins,
                  trainSampleRatio,m_channels,shuffle);
    m_Q = m_trainImgs.size();
}

void labelBin2Names(std::vector<std::vector<bool>> labelBins, std::vector<std::string> labelStrings, 
                    std::vector<std::string> &names)
{
    for(int i=0;i<labelBins.size();i++)
    {
        for(int j=0;j<labelBins[i].size();j++)
        {
            if(labelBins[i][j])
            {
                names.push_back(labelStrings[j]);
                break;
            }
        }
    }
}

void ELM_IN_ELM_Model::loadStandardFaceDataset(const std::string path, const float trainSampleRatio, 
                                               const int resizeWidth, const int resizeHeight, bool shuffle)
{
    m_width = resizeWidth;
    m_height = resizeHeight;
    m_channels = 1;
    
    inputImgsFrom(path,m_label_string,m_trainImgs,
                  m_testImgs,m_trainLabelBins,m_testLabelBins,
                  trainSampleRatio,m_channels,shuffle);
    
    std::vector<std::string> names_train;
    labelBin2Names(m_trainLabelBins,m_label_string,names_train);
    faceImgPreprocessing(m_trainImgs,m_faceFeats,names_train);
    std::vector<std::string> names_test;
    labelBin2Names(m_trainLabelBins,m_label_string,names_test);
    faceImgPreprocessing(m_testImgs,m_faceFeats_test,names_test);
    
    //test
    for(int i=0;i<m_faceFeats.rows;i++)
    {
        std::cout<<"name:"<<names_train[i]<<std::endl;
        std::cout<<"resnet feat:\n"<<m_faceFeats.rowRange(i,i+1)<<std::endl;
    }
    
    m_C = m_label_string.size();
    m_Q = m_trainImgs.size();
}

void ELM_IN_ELM_Model::loadMnistData(const std::string path, const float trainSampleRatio, bool shuffle)
{
    loadMnistData_csv(path,trainSampleRatio,
                      m_trainImgs,m_testImgs,m_trainLabelBins,m_testLabelBins,shuffle);
    
    m_Q = m_trainImgs.size();
    
    m_width = 28;
    m_height = 28;
    m_channels = 1;
}

void ELM_IN_ELM_Model::fitSubModels_faceFeat(int batchSize, bool validating, bool verbose)
{
    m_subModelToTrain.inputData_1d(m_faceFeats,m_trainLabelBins);
    m_subModelToTrain.inputData_1d_test(m_faceFeats_test,m_testLabelBins);
    
    int randomState = (unsigned)time(nullptr);
    
    //训练子模型
    for(int i=0;i<m_n_models;i++)
    {
        if(m_subModelHiddenNodes[i] != -1)
            m_subModelToTrain.setHiddenNodes(m_subModelHiddenNodes[i]);
        m_subModelToTrain.setRandomState(randomState++);
        m_subModelToTrain.fit(batchSize, validating,verbose);
        m_subModelToTrain.save(m_modelPath+"subModel"+std::to_string(i)+".xml",
                               m_modelPath+"subK"+std::to_string(i)+".xml");
        
        m_subModelToTrain.clear();
    }
}

void ELM_IN_ELM_Model::fitSubModels(int batchSize, bool validating, bool verbose)
{
    m_subModelToTrain.inputData_2d(m_trainImgs,m_trainLabelBins,m_width,m_height,m_channels);
    m_subModelToTrain.inputData_2d_test(m_testImgs,m_testLabelBins);
    
    int randomState = (unsigned)time(nullptr);
    
    //训练子模型
    for(int i=0;i<m_n_models;i++)
    {
        if(m_subModelHiddenNodes[i] != -1)
            m_subModelToTrain.setHiddenNodes(m_subModelHiddenNodes[i]);
        m_subModelToTrain.setRandomState(randomState++);
        m_subModelToTrain.fit(batchSize, validating,verbose);
        m_subModelToTrain.save(m_modelPath+"subModel"+std::to_string(i)+".xml",
                               m_modelPath+"subK"+std::to_string(i)+".xml");
        
        m_subModelToTrain.clear();
    }
    
}

void ELM_IN_ELM_Model::fitMainModel(int batchSize, bool validating, bool verbose)
{
    std::cout<<"【elm-in-elm训练开始】--------------------------------------"<<std::endl;
    
    if(m_trainImgs.empty())
        return;
    
    //载入子模型
    if(m_subModels.empty())
    {
        m_subModels.resize(m_n_models);
        for(int i=0;i<m_n_models;i++)
            m_subModels[i].load(m_modelPath+"subModel"+std::to_string(i)+".xml",
                              m_modelPath+"subK"+std::to_string(i)+".xml");
    }
    
    //为H和T分配空间
    int M = m_n_models;
    if(batchSize==-1)
        batchSize = m_trainImgs.size();
    m_C = m_label_string.size();
    cv::Mat H(cv::Size(M*m_C,batchSize),CV_32F);
    cv::Mat batchTarget(cv::Size(m_C,batchSize),CV_32F);
    
    //转化标签 bool->Mat
    cv::Mat allTarget;
    label2target(m_trainLabelBins,allTarget);
    
    //输出矩阵大小信息
    if(verbose)
    {
        std::cout<<"Q: "<<m_Q<<std::endl
                 <<"batchSize: "<<batchSize<<std::endl
                 <<"M: "<<M<<std::endl
                 <<"C: "<<m_C<<std::endl;
    }
    
    //m_K的初始化
    if(m_K.empty())
    {
        m_K.create(cv::Size(M*m_C,M*m_C),CV_32F);
        m_K = cv::Scalar(0);
    }
    
    //m_F的初始化
    if(m_F.empty())
    {
        m_F.create(cv::Size(m_C,M*m_C),CV_32F);
        m_F = cv::Scalar(0);
    }
    
    int trainedRatio = 0;
    for(int i=0;i+batchSize<=m_Q;i+=batchSize)
    {
        std::vector<cv::Mat> batchTrainImgs(m_trainImgs.begin()+i,m_trainImgs.begin()+i+batchSize);

        //为H和batchTarget赋值
        for(int m=0;m<m_n_models;m++)
        {
            cv::Mat ROI = H(cv::Range(0,batchSize),cv::Range(m*m_C,(m+1)*m_C));
            m_subModels[m].batchQuery(batchTrainImgs,ROI);
        }
        batchTarget = allTarget(cv::Range(i,i+batchSize),cv::Range(0,m_C));

        //迭代更新K
        m_K = m_K + H.t() * H;
        
        //迭代更新F
        m_F = m_F + m_K.inv(1) * H.t() * (batchTarget - H*m_F);

        //输出信息
        if(verbose)
        {
            int ratio = (i+batchSize)/(float)m_Q*100;
            if( ratio - trainedRatio >= 1)
            {
                trainedRatio = ratio;
                
                //计算在该批次训练数据上的准确率
                cv::Mat output = H * m_F;
                float score = calcScore(output,batchTarget);
                std::cout<<"Score on batch training data:"<<score<<std::endl;
                
                //计算在测试数据上的准确率
                if(validating && m_testImgs.size()>0)
                    validate();
            }
        }
    }
    
    std::cout<<"【elm-in-elm训练结束】--------------------------------------"<<std::endl;
    
/*std::cout<<"T:"<<T.size<<"\n"<<T<<std::endl;
std::cout<<"H:"<<H.size<<"\n"<<H<<std::endl;
std::cout<<"F:"<<m_F.size<<"\n"<<m_F<<std::endl;
std::cout<<"H*F:"<<realOutput.size<<"\n"<<H*m_F<<std::endl;
*/
}

void ELM_IN_ELM_Model::fitMainModel_faceFeat(int batchSize, bool validating, bool verbose)
{
    std::cout<<"【elm-in-elm训练开始】--------------------------------------"<<std::endl;
    
    if(m_faceFeats.empty())
        return;
    
    m_Q = m_faceFeats.rows;
    
    //载入子模型
    if(m_subModels.empty())
    {
        m_subModels.resize(m_n_models);
        for(int i=0;i<m_n_models;i++)
        {
            m_subModels[i] = ELM_Model();
            
            m_subModels[i].load(m_modelPath+"subModel"+std::to_string(i)+".xml",
                              m_modelPath+"subK"+std::to_string(i)+".xml");
        }
    }
    
    //为H和T分配空间
    int M = m_n_models;
    if(batchSize==-1)
        batchSize = m_faceFeats.rows;
    m_C = m_label_string.size();
    cv::Mat H(cv::Size(M*m_C,batchSize),CV_32F);
    cv::Mat batchTarget(cv::Size(m_C,batchSize),CV_32F);
    
    //转化标签 bool->Mat
    cv::Mat allTarget;
    label2target(m_trainLabelBins,allTarget);
    
    //输出矩阵大小信息
    if(verbose)
    {
        std::cout<<"Q: "<<m_Q<<std::endl
                 <<"batchSize: "<<batchSize<<std::endl
                 <<"M: "<<M<<std::endl
                 <<"C: "<<m_C<<std::endl;
    }
    
    //m_K的初始化
    if(m_K.empty())
    {
        m_K.create(cv::Size(M*m_C,M*m_C),CV_32F);
        m_K = cv::Scalar(0);
    }
    
    //m_F的初始化
    if(m_F.empty())
    {
        m_F.create(cv::Size(m_C,M*m_C),CV_32F);
        m_F = cv::Scalar(0);
    }
    
    int trainedRatio = 0;
    for(int i=0;i+batchSize<=m_Q;i+=batchSize)
    {
        cv::Mat batchTrainFeats = m_faceFeats.rowRange(i,i+batchSize);
        
        //为H和batchTarget赋值
        for(int m=0;m<m_n_models;m++)
        {
            cv::Mat ROI = H(cv::Range(0,batchSize),cv::Range(m*m_C,(m+1)*m_C));
            m_subModels[m].batchQueryFeats(batchTrainFeats,ROI);
            //std::cout<<"subModel["<<m<<"]:\n"<<ROI<<std::endl;
            //std::cout<<"subModel["<<m<<"]:\n"<<binMat(ROI)<<std::endl;
            binMat(ROI).copyTo(ROI);
        }
        batchTarget = allTarget(cv::Range(i,i+batchSize),cv::Range(0,m_C));
        
        //迭代更新K
        m_K = m_K + H.t() * H;
        
        //迭代更新F
        m_F = m_F + m_K.inv(1) * H.t() * (batchTarget - H*m_F);

        //输出信息
        if(verbose)
        {
            int ratio = (i+batchSize)/(float)m_Q*100;
            if( ratio - trainedRatio >= 1)
            {
                trainedRatio = ratio;
                
                //计算在该批次训练数据上的准确率
                cv::Mat output = H * m_F;
                //std::cout<<"mainModel:\n"<<output<<std::endl;
                //std::cout<<"mainModel:\n"<<binMat(output)<<std::endl;
                float score = calcScore(output,batchTarget);
                std::cout<<"Score on batch training data:"<<score<<std::endl;
                
                /*
                //计算在测试数据上的准确率
                if(validating && m_testImgs.size()>0)
                    validate();
                */
            }
        }
    }
    
    std::cout<<"【elm-in-elm训练结束】--------------------------------------"<<std::endl;
    
/*std::cout<<"T:"<<T.size<<"\n"<<T<<std::endl;
std::cout<<"H:"<<H.size<<"\n"<<H<<std::endl;
std::cout<<"F:"<<m_F.size<<"\n"<<m_F<<std::endl;
std::cout<<"H*F:"<<realOutput.size<<"\n"<<H*m_F<<std::endl;
*/
}

float ELM_IN_ELM_Model::validate()
{
    int M = m_n_models;
    
    cv::Mat H_test(cv::Size(M*m_C,m_testImgs.size()),CV_32F);
    cv::Mat T_test(cv::Size(m_C,m_testImgs.size()),CV_32F);
    
    //给H_test和T_test赋值
    for(int m=0;m<m_n_models;m++)
    {
        cv::Mat ROI = H_test(cv::Range(0,m_testImgs.size()),cv::Range(m*m_C,(m+1)*m_C));
        m_subModels[m].batchQuery(m_testImgs,ROI);
    }
    label2target(m_testLabelBins,T_test);
    
    //计算
    cv::Mat output = H_test * m_F;
    float finalScore_test = calcScore(output,T_test);
    
    std::cout<<"Score on validation data:"<<finalScore_test<<std::endl;
    return finalScore_test;
}

void ELM_IN_ELM_Model::save()
{
    if(m_F.empty())
        return;
    
    cv::FileStorage fswrite(m_modelPath+"mainModel.xml",cv::FileStorage::WRITE);
    
    fswrite<<"n_models"<<m_n_models;
    fswrite<<"subModelPath"<<m_modelPath;
    fswrite<<"width"<<m_width;
    fswrite<<"height"<<m_height;
    fswrite<<"channels"<<m_channels;
    fswrite<<"C"<<m_C;
    fswrite<<"F"<<m_F;
    fswrite<<"label_string"<<m_label_string;
    
    fswrite.release();
    
    cv::FileStorage K_fswrite(m_modelPath+"mainK.xml",cv::FileStorage::WRITE);
    K_fswrite<<"K"<<m_K;
    K_fswrite.release();
}

void ELM_IN_ELM_Model::load(std::string modelDir)
{
    m_modelPath = modelDir;
    if(m_modelPath[m_modelPath.length()-1] != '/')
        m_modelPath.append("/");
    
    cv::FileStorage fsread(m_modelPath+"mainModel.xml",cv::FileStorage::READ);

    fsread["n_models"]>>m_n_models;
    fsread["subModelPath"]>>m_modelPath;
    setInitPara(m_n_models,m_modelPath);
    fsread["channels"]>>m_channels;
    fsread["width"]>>m_width;
    fsread["height"]>>m_height;
    fsread["C"]>>m_C;
    fsread["F"]>>m_F;
    fsread["label_string"]>>m_label_string;

    fsread.release();
    
    cv::FileStorage K_fsread(m_modelPath+"mainK.xml",cv::FileStorage::READ);
    K_fsread["K"]>>m_K;
    K_fsread.release();

    //加载子模型
    m_subModels.resize(m_n_models);
    for(int m=0;m<m_n_models;m++)
    {
        m_subModels[m] = ELM_Model();
        
        m_subModels[m].load(m_modelPath+"subModel"+std::to_string(m)+".xml",
                            m_modelPath+"subK"+std::to_string(m)+".xml");
    }
    
}

void ELM_IN_ELM_Model::query(const cv::Mat &mat, std::string &label)
{
    cv::Mat H(cv::Size(m_n_models*m_C,1),CV_32F);
    
    for(int m=0;m<m_n_models;m++)
    {
        cv::Mat ROI = H(cv::Range(0,1),cv::Range(m*m_C,(m+1)*m_C));
        m_subModels[m].query(mat,ROI);
    }
    
    cv::Mat output = H * m_F;
    //std::cout<<output<<std::endl;
    int maxId = getMaxId(output);
    
    label.assign(m_label_string[maxId]);
}

void ELM_IN_ELM_Model::queryFace(const cv::Mat &mat, std::string &label)
{
    cv::Mat feat;
    faceImgPreprocessing(mat,feat);
    
    cv::Mat H(cv::Size(m_n_models*m_C,1),CV_32F);
    
    for(int m=0;m<m_n_models;m++)
    {
        cv::Mat ROI = H(cv::Range(0,1),cv::Range(m*m_C,(m+1)*m_C));
        m_subModels[m].queryFeat(mat,ROI);
        binMat(ROI).copyTo(ROI);
    }
    
    cv::Mat output = H * m_F;
    int maxId = getMaxId(output);
    
    label.assign(m_label_string[maxId]);
}

void ELM_IN_ELM_Model::query(const cv::Mat &mat, int n, std::vector<std::string> &names)
{
    if(m_F.empty())
        return;
    
    if(n>m_C)
        n = m_C;
    
    cv::Mat H(cv::Size(m_n_models*m_C,1),CV_32F);
    
    for(int m=0;m<m_n_models;m++)
    {
        cv::Mat ROI = H(cv::Range(0,1),cv::Range(m*m_C,(m+1)*m_C));
        m_subModels[m].query(mat,ROI);
    }
    
    cv::Mat output = H * m_F;
    
    std::vector<int> maxIds;
    getMaxNId(output,n,maxIds);
    
    for(size_t i=0;i<n;i++)
        names.push_back(m_label_string[maxIds[i]]);
}

void ELM_IN_ELM_Model::queryFace(const cv::Mat &feat, int n, std::vector<std::string> &names)
{
    if(m_F.empty())
        return;
    
    if(n>m_C)
        n = m_C;
    
    cv::Mat H(cv::Size(m_n_models*m_C,1),CV_32F);
    
    for(int m=0;m<m_n_models;m++)
    {
        cv::Mat ROI = H(cv::Range(0,1),cv::Range(m*m_C,(m+1)*m_C));
        m_subModels[m].queryFeat(feat,ROI);
        //std::cout<<"subModel["<<m<<"]:\n"<<ROI/*binMat(ROI)*/<<std::endl;
        //std::cout<<"subModel["<<m<<"]:\n"<<binMat(ROI)<<std::endl;
        binMat(ROI).copyTo(ROI);
    }
    
    cv::Mat output = H * m_F;
    //std::cout<<"mainModel:\n"<<output/*binMat(output)*/<<std::endl;
    
    //test
    //for(int i=0;i<output.cols;i++)
    //    std::cout<<m_label_string[i]<<":"<<output.at<float>(0,i)<<std::endl;
    
    std::vector<int> maxIds;
    getMaxNId(output,n,maxIds);
    
    for(size_t i=0;i<n;i++)
        names.push_back(m_label_string[maxIds[i]]);
}

void ELM_IN_ELM_Model::queryFace(const cv::Mat &feat, int n, std::vector<std::string> &names, std::vector<float> &sims)
{
    if(m_F.empty())
        return;
    
    if(n>m_C)
        n = m_C;
    
    cv::Mat H(cv::Size(m_n_models*m_C,1),CV_32F);
    
    for(int m=0;m<m_n_models;m++)
    {
        cv::Mat ROI = H(cv::Range(0,1),cv::Range(m*m_C,(m+1)*m_C));
        m_subModels[m].queryFeat(feat,ROI);
        binMat(ROI).copyTo(ROI);
    }
    
    cv::Mat output = H * m_F;
    
    std::vector<int> maxIds;
    getMaxNId(output,n,maxIds);
    
    for(size_t i=0;i<n;i++)
    {
        names.push_back(m_label_string[maxIds[i]]);
        sims.push_back(output.at<float>(0,maxIds[i]));
    }
}

void ELM_IN_ELM_Model::clearTrainData()
{
    if(m_trainImgs.empty())
        return;
    
    m_subModelToTrain.clearTrainData();
    for(int i=0;i<m_subModels.size();i++)
        m_subModels[i].clearTrainData();
    
    if(!m_trainImgs.empty())
        m_trainImgs.clear();
    if(!m_trainLabelBins.empty())
        m_trainLabelBins.clear();
    if(!m_testImgs.empty())
        m_testImgs.clear();
    if(!m_testLabelBins.empty())
        m_testLabelBins.clear();
}

void ELM_IN_ELM_Model::clearTrainData_feat()
{
    if(m_faceFeats.empty())
        return;
    
    m_subModelToTrain.clearTrainData();
    for(int i=0;i<m_subModels.size();i++)
        m_subModels[i].clearTrainData();
    
    if(!m_faceFeats.empty())
        m_faceFeats = cv::Mat();
    if(!m_trainLabelBins.empty())
        m_trainLabelBins.clear();
    if(!m_faceFeats_test.empty())
        m_faceFeats_test = cv::Mat();
    if(!m_testLabelBins.empty())
        m_testLabelBins.clear();
}

void ELM_IN_ELM_Model::trainNewImg(const cv::Mat &img, const std::string label)
{
    clearTrainData();
    m_trainImgs.push_back(img);
    std::vector<bool> labelBin(m_C,0);
    for(int i=0;i<m_label_string.size();i++)
        if(label == m_label_string[i])
        {
            labelBin[i] = 1;
            break;
        }
    m_trainLabelBins.push_back(labelBin);
    m_Q = 1;
    
    for(int i=0;i<m_n_models;i++)
    {
        m_subModelToTrain = ELM_Model();
        
        m_subModelToTrain.load(m_modelPath+"subModel"+std::to_string(i)+".xml",
                          m_modelPath+"subK"+std::to_string(i)+".xml");
        
        m_subModelToTrain.trainNewImg(img,label);
        m_subModelToTrain.save(m_modelPath+"subModel"+std::to_string(i)+".xml",
                               m_modelPath+"subK"+std::to_string(i)+".xml");
    }
    
    fitMainModel(-1,false,false);
}

void ELM_IN_ELM_Model::trainNewFace(const cv::Mat &img, const std::string label)
{
    cv::Mat feat;
    faceImgPreprocessing(img,feat,label);
    
    clearTrainData_feat();
    m_faceFeats.push_back(feat);
    
    //std::cout<<"name"<<label<<std::endl;
    //std::cout<<"feat:\n"<<feat<<std::endl;
    
    std::vector<bool> labelBin(m_C,0);
    for(int i=0;i<m_label_string.size();i++)
        if(label == m_label_string[i])
        {
            labelBin[i] = 1;
            break;
        }
    m_trainLabelBins.push_back(labelBin);
    m_Q = 1;
    
    for(int i=0;i<m_n_models;i++)
    {
        m_subModelToTrain = ELM_Model();
        
        m_subModelToTrain.load(m_modelPath+"subModel"+std::to_string(i)+".xml",
                          m_modelPath+"subK"+std::to_string(i)+".xml");
        
        m_subModelToTrain.trainNewFace(feat,labelBin);
        
        m_subModelToTrain.save(m_modelPath+"subModel"+std::to_string(i)+".xml",
                               m_modelPath+"subK"+std::to_string(i)+".xml");
    }
    fitMainModel_faceFeat(-1,false);
}

bool ELM_IN_ELM_Model::isEmpty()
{
    return m_F.empty();
}
