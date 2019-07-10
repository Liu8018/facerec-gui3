#ifndef FUNCS_H
#define FUNCS_H

#include <opencv2/core.hpp>

//加载图像
void inputImgsFrom(const std::string datasetPath, 
                   std::vector<std::string> &label_string,
                   std::vector<cv::Mat> &trainImgs, std::vector<cv::Mat> &testImgs, 
                   std::vector<std::vector<bool> > &trainLabelBins, 
                   std::vector<std::vector<bool> > &testLabelBins, 
                   const float trainSampleRatio, const int channels, 
                   bool shuffle=true);

//加载mnist数据集
void loadMnistData_csv(const std::string path, const float trainSampleRatio, 
                   std::vector<cv::Mat> &trainImgs, std::vector<cv::Mat> &testImgs, 
                   std::vector<std::vector<bool> > &trainLabelBins, 
                   std::vector<std::vector<bool> > &testLabelBins, 
                   bool shuffle=true);

//二维数据转换为一维(从AxB到1xAB)
void mat2line(const cv::Mat &mat, cv::Mat &line, const int channels);
void mats2lines(const std::vector<cv::Mat> &mats, cv::Mat &output, const int channels);

//转化label为target(从QxC到QxC)
void label2target(const std::vector<std::vector<bool>> &labels, cv::Mat &target);

//随机生成矩阵
void randomGenerate(cv::Mat &mat, cv::Size size, int randomState=-1);

//激活
void activate(cv::Mat &H, const std::string method);

//加偏置
void addBias(cv::Mat &mat, const cv::Mat &bias);

//激活函数sigmoid
void sigmoid(cv::Mat &H);

//归一化
void normalize(cv::Mat &mat);
void normalize_img(cv::Mat &mat);

//遍历一个目录
void traverseFile(const std::string directory, std::vector<std::string> &files);

//找最大值
int getMaxId(const cv::Mat &line);
void getMaxNId(const cv::Mat &line, int n, std::vector<int> &ids);

//计分
float calcScore(const cv::Mat &outputData, const cv::Mat &target);

cv::Mat binMat(const cv::Mat &mat);

#endif // FUNCS_H
