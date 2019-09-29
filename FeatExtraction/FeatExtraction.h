#ifndef FEATEXTRACTION_H
#define FEATEXTRACTION_H

#include <dlib/dnn.h>
#include <opencv2/core.hpp>

//定义好一堆模板别名，以供后续方便使用
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N,BN,1,dlib::tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2,2,2,2,dlib::skip1<dlib::tag2<block<N,BN,2,dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<dlib::con<N,3,3,1,1,dlib::relu<BN<dlib::con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = dlib::relu<residual<block,N,dlib::affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block,N,dlib::affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = dlib::loss_metric<dlib::fc_no_bias<128,dlib::avg_pool_everything<
                            alevel0<alevel1<alevel2<alevel3<alevel4<
                            dlib::max_pool<3,3,2,2,dlib::relu<dlib::affine<dlib::con<32,7,7,2,2,
                            dlib::input_rgb_image_sized<150>>>>>>>>>>>>>;

class FeatExtraction
{
public:
    FeatExtraction();
    
    void extract(const cv::Mat &img, cv::Mat &feat);
    void extract_batch(const std::vector<cv::Mat> &imgs, cv::Mat &feats);
    
    void getShape(const cv::Mat &inputImg, const cv::Rect &faceRect, dlib::full_object_detection &shape);
    void alignFace(const cv::Mat &inputImg, cv::Rect &faceRect, cv::Mat &resultImg);
    bool judgeFaceAndAlign(const cv::Mat &inputImg, cv::Rect &faceRect, cv::Mat &resultImg);
    
    void saveFeat_add(std::string name, const cv::Mat &feat);
    void saveFeats_overwrite(std::vector<std::string> names, const cv::Mat &feats);
    void loadFeats(cv::Mat &feats, std::vector<std::string> &names);
    void loadFeats(const std::vector<std::string> &candidates, cv::Mat &feats, std::vector<std::string> &names);
    void saveFeats_resnet(std::vector<std::string> names, const cv::Mat &feats);
    void loadFeats_resnet(std::vector<std::string> &names, cv::Mat &feats);
    void addResnetFeat(const cv::Mat &faceImg, const std::string &name);
    
    //resnet方法
    void resnetEx(const cv::Mat &faceMat, cv::Mat &feat);
    
    //高维lbp方法
    void highDimLbpEx(const cv::Mat &faceMat, cv::Mat &feat);
    
    //提取多种特征
    void multiFeatEx(const cv::Mat &faceMat, cv::Mat &feat);
    
    //pca
    void pcaEx(const cv::Mat &faceMat, cv::Mat &feat);
    void calcPCA();
    void updatePCA(const cv::Mat &newFace);
    void loadPCA();
    void savePCA();
    
    //计算最高相似度
    float getMaxSim(const cv::Mat &feat, std::string name);
    
private:
    std::string m_method;
    dlib::shape_predictor m_shapePredictor;
    
    anet_type m_resnet;
    
    cv::Mat m_PCA_mean;
    cv::Mat m_PCA_eigenVecs;
    cv::Mat m_PCA_DTD;
    cv::PCA m_pca;
    
    void dlibPoint2cvPoint(const dlib::full_object_detection &S, std::vector<cv::Point> &L);
    void dlibPoint2cvPoint2f(const dlib::full_object_detection &S, std::vector<cv::Point2f> &L);
    void cvRect2dlibRect(const cv::Rect &cvRec, dlib::rectangle &dlibRec);
    void drawShape(cv::Mat &img, dlib::full_object_detection shape);
    
    cv::Mat padImg(const cv::Mat &img, float paddingRatio);
    
    void getNbBlock(const cv::Mat &img, cv::Point center, int blockSize, cv::Mat &block);
};

extern FeatExtraction g_featEX;

#endif // FEATEXTRACTION_H
