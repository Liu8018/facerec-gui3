#include "ELM_functions.h"
#include <sys/stat.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <dirent.h>
#include <algorithm>
#include <iostream>
#include <fstream>

//加载图像
void inputImgsFrom(const std::string datasetPath, 
                   std::vector<std::string> &label_string, 
                   std::vector<cv::Mat> &trainImgs, std::vector<cv::Mat> &testImgs, 
                   std::vector<std::vector<bool> > &trainLabelBins, 
                   std::vector<std::vector<bool> > &testLabelBins, 
                   const float trainSampleRatio, const int channels, bool shuffle)
{
    std::vector<std::string> srcFiles;
    traverseFile(datasetPath,srcFiles);
    
    std::vector<std::string> files;
    
    for(int i=0;i<srcFiles.size();i++)
    {
        //判断是否是文件夹,不是就跳过
        char const*path = srcFiles[i].data();
        struct stat s_buf;
        stat(path,&s_buf);
        if(!S_ISDIR(s_buf.st_mode))
            continue;
        
        files.push_back(srcFiles[i]);
    }
    
    int classes = files.size();
    
    for(int i=0;i<files.size();i++)
    {
        std::cout<<"[INFO] loading data from "<<files[i]<<std::endl;
        
        std::vector<std::string> subdir_files;
        traverseFile(files[i],subdir_files);
        
        //随机打乱顺序
        srand(time(nullptr));
        if(shuffle)
            std::random_shuffle(subdir_files.begin(),subdir_files.end());
        
        if(!subdir_files.empty())
        {
            std::string label = files[i].substr(files[i].find_last_of('/')+1,files[i].length()-1);
            label_string.push_back(label);
            
            int trainSamples = subdir_files.size()*trainSampleRatio;
            
            for(int j=0;j<trainSamples;j++)
            {
                cv::Mat src;
                if(channels == 3)
                    src = cv::imread(subdir_files[j]);
                if(channels == 1)
                    src = cv::imread(subdir_files[j],0);
                
                trainImgs.push_back(src);
                
                std::vector<bool> labelBin(classes,0);
                labelBin[i] = 1;
                trainLabelBins.push_back(labelBin);
            }
            
            for(int j=trainSamples;j<subdir_files.size();j++)
            {
                cv::Mat src;
                if(channels == 3)
                    src = cv::imread(subdir_files[j]);
                if(channels == 1)
                    src = cv::imread(subdir_files[j],0);
                
                testImgs.push_back(src);
                
                std::vector<bool> labelBin(classes,0);
                labelBin[i] = 1;
                testLabelBins.push_back(labelBin);
            }
        }
    }
}

void loadMnistData_csv(const std::string path, const float trainSampleRatio,
                   std::vector<cv::Mat> &trainImgs, std::vector<cv::Mat> &testImgs, 
                   std::vector<std::vector<bool> > &trainLabelBins, 
                   std::vector<std::vector<bool> > &testLabelBins, bool shuffle)
{
    std::ifstream fin(path);
    
    std::string tmpLine;
    std::vector<std::string> lines;
    
    while(std::getline(fin,tmpLine))
        lines.push_back(tmpLine);
    
    srand(time(nullptr));
    if(shuffle)
        std::random_shuffle(lines.begin(),lines.end());
    
    int trainSize = lines.size()*trainSampleRatio;
    
    for(int j=0;j<trainSize;j++)
    {
        std::string line;
        line.assign(lines[j]);
        
        std::vector<bool> label_bin(10,0);
        label_bin[line[0]-48] = 1;
        trainLabelBins.push_back(label_bin);

        cv::Mat img(28,28,CV_8U);
        int pixNum=0;
        for(int i=2;i<line.size();i++)
        {
            int value=0;
            
            if(line[i] == ',')
                continue;
            
            while(i<line.size() && line[i] != ',')
            {
                value = value*10 + line[i] - 48;
                i++;
            }
            i--;
            
            int y = pixNum/28;
            int x = pixNum%28;
            img.at<uchar>(y,x) = value;
            
            pixNum++;
        }
        
        trainImgs.push_back(img);
    }
    
    for(int j=trainSize;j<lines.size();j++)
    {
        std::string line;
        line.assign(lines[j]);
        
        std::vector<bool> label_bin(10,0);
        label_bin[line[0]-48] = 1;
        testLabelBins.push_back(label_bin);

        cv::Mat img(28,28,CV_8U);
        int pixNum=0;
        for(int i=2;i<line.size();i++)
        {
            int value=0;
            
            if(line[i] == ',')
                continue;
            
            while(i<line.size() && line[i] != ',')
            {
                value = value*10 + line[i] - 48;
                i++;
            }
            i--;
            
            int y = pixNum/28;
            int x = pixNum%28;
            img.at<uchar>(y,x) = value;
            
            pixNum++;
        }
        
        testImgs.push_back(img);
    }
}

//二维数据转换为一维
//从AxB到1xAB
void mat2line(const cv::Mat &mat, cv::Mat &line, const int channels)
{
    cv::resize(line,line,cv::Size(mat.rows*mat.cols*channels,1));
    
    if(channels==1)
    {
        for(int r=0;r<mat.rows;r++)
            for(int c=0;c<mat.cols;c++)
                line.at<float>(0,r*mat.cols+c) = float(mat.at<uchar>(r,c));
    }
    if(channels==3)
    {
        std::vector<cv::Mat> channels;
        cv::split(mat,channels);
        int j=0;
        for(int i=0;i<3;i++)
            for(int r=0;r<channels[i].rows;r++)
                for(int c=0;c<channels[i].cols;c++)
                {
                    line.at<float>(0,j) = float(channels[i].at<uchar>(r,c));
                    j++;
                }
    }
}

void mats2lines(const std::vector<cv::Mat> &mats, cv::Mat &output, const int channels)
{
    if(mats.empty())
        return;
    
    output.create(cv::Size(mats[0].rows*mats[0].cols*channels,mats.size()),CV_32F);
    
    for(int i=0;i<mats.size();i++)
    {
        cv::Mat lineROI = output(cv::Range(i,i+1),cv::Range(0,output.cols));
        mat2line(mats[i],lineROI, channels);
    }
};

//转化label为target
//从QxC到QxC
void label2target(const std::vector<std::vector<bool>> &labels, cv::Mat &target)
{
    if(labels.empty())
        return;
    
    int labelLength = labels[0].size();
    target.create(cv::Size(labelLength,labels.size()),CV_32F);
    for(int i=0;i<labels.size();i++)
    {
        for(int j=0;j<labelLength;j++)
            target.at<float>(i,j) = float(labels[i][j]);
    }
}

void addBias(cv::Mat &mat, const cv::Mat &bias)
{
    for(int i=0;i<mat.rows;i++)
        for(int j=0;j<mat.cols;j++)
            mat.at<float>(i,j) += bias.at<float>(0,j);
}

void activate(cv::Mat &H, const std::string method)
{
    if(method == "sigmoid")
        sigmoid(H);
}

//sigmoid激活函数
void sigmoid(cv::Mat &H)
{
    for(int i=0;i<H.rows;i++)
        for(int j=0;j<H.cols;j++)
            H.at<float>(i,j) = 1.0 / ( 1.0 + std::exp(-H.at<float>(i,j)) );
}

//归一化(0-1)
void normalize(cv::Mat &mat)
{
    double minVal,maxVal;
    cv::minMaxIdx(mat,&minVal,&maxVal);
    
    for(int i=0;i<mat.rows;i++)
        for(int j=0;j<mat.cols;j++)
            mat.at<float>(i,j) = (mat.at<float>(i,j)-minVal) / (maxVal-minVal);
}

void normalize_img(cv::Mat &mat)
{
    for(int i=0;i<mat.rows;i++)
        for(int j=0;j<mat.cols;j++)
            mat.at<float>(i,j) = mat.at<float>(i,j) / 255.0;
}

//遍历一个目录
void traverseFile(const std::string directory, std::vector<std::string> &files)
{
    std::string prefix = directory;
    if(directory[directory.length()-1] != '/')
        prefix += '/';
    
    files.clear();
    
    const char * char_dir = directory.data();
    
    DIR* dir = opendir(char_dir);//打开指定目录
    dirent* p = nullptr;//定义遍历指针
    while((p = readdir(dir)) != nullptr)//开始逐个遍历
    {
        //linux平台下目录中有"."和".."隐藏文件，需要过滤掉
        if(p->d_name[0] != '.')//d_name是一个char数组，存放当前遍历到的文件名
        {
            std::string name = prefix + std::string(p->d_name);
            files.push_back(name);
        }
    }
    closedir(dir);//关闭指定目录
}

int getMaxId(const cv::Mat &line)
{
    double minVal,maxVal;
    int minIdx[2],maxIdx[2];
    
    cv::minMaxIdx(line,&minVal,&maxVal,minIdx,maxIdx);
    
    return maxIdx[1];
}

int getMinId(const cv::Mat &line)
{
    double minVal,maxVal;
    int minIdx[2],maxIdx[2];
    
    cv::minMaxIdx(line,&minVal,&maxVal,minIdx,maxIdx);
    
    return minIdx[1];
}

void getMaxNId(const cv::Mat &line, int n, std::vector<int> &ids)
{
    cv::Mat tmpLine = line.clone();
    
    ids.resize(n);
    
    float minVal = line.at<float>(0,getMinId(line));
    
    for(int i=0;i<n;i++)
    {
        ids[i] = getMaxId(tmpLine);
        
        tmpLine.at<float>(0,ids[i]) = minVal;
    }
}

float calcScore(const cv::Mat &outputData, const cv::Mat &target)
{
    int score = 0;
    for(int i=0;i<outputData.rows;i++)
    {
        cv::Mat ROI_o = outputData(cv::Range(i,i+1),cv::Range(0,outputData.cols));
        int maxId_O = getMaxId(ROI_o);
        
        cv::Mat ROI_t = target(cv::Range(i,i+1),cv::Range(0,target.cols));
        int maxId_T = getMaxId(ROI_t);
        
        if(maxId_O == maxId_T)
            score++;
    }
std::cout<<"score:"<<score<<std::endl;
std::cout<<"size:"<<outputData.rows<<std::endl;
    float finalScore = score/(float)outputData.rows;
    
    return finalScore;
}

void randomGenerate(cv::Mat &mat, cv::Size size, int randomState)
{
    mat.create(size,CV_32F);
    
    cv::RNG rng;
    if(randomState != -1)
        rng.state = randomState;
    else
        rng.state = (unsigned)time(nullptr);
    for(int i=0;i<mat.rows;i++)
        for(int j=0;j<mat.cols;j++)
            mat.at<float>(i,j) = rng.uniform(-1.0,1.0);
}

