#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <opencv2/core.hpp>
#include <map>

//判断是否空的人脸库
bool isEmptyFaceDb();

//升级resnet数据库
void updateResnetDb();

//重新训练elm-in-elm模型
void refitEIEModel();

//升级数据库
void handleFaceDb(int method);

//给库中检测并裁剪过的人脸图片打上标记
void markImg(cv::Mat &img);

//检测图片是否已被打上标记
bool isMarkedImg(const cv::Mat &img);

//获取标准文件夹格式下的文件
void getFiles(std::string path, std::map<std::string, std::string> &files);
void getFiles_less(std::string path, std::map<std::string, std::string> &files);
void getFileByName(std::string path, std::vector<cv::Mat> &imgs);
void getSubDirs(std::string path, std::vector<std::string> &subDirs);

//均衡化
void equalizeIntensity(cv::Mat &img);

//存储/读取xml文件
void saveMatAsXml(const cv::Mat &mat, std::string path);
void loadXmlAsMat(std::string path, cv::Mat &mat);

//脸部图像预处理
void faceImgPreprocessing(const cv::Mat &img, cv::Mat &feat);
void faceImgPreprocessing(const std::vector<cv::Mat> &imgs, cv::Mat &feats);
void faceImgPreprocessing(const cv::Mat &img, cv::Mat &feat, std::string name);
void faceImgPreprocessing(const std::vector<cv::Mat> &imgs, cv::Mat &feats, std::vector<std::string> names);


#endif // FUNCTIONS_H
