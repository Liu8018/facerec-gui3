#include "functions.h"
#include "FaceDetection/FaceDetection.h"
#include "FeatExtraction/FeatExtraction.h"
#include "elm/elm_in_elm_model.h"
#include <dlib/opencv/cv_image.h>
#include "params.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "elm/ELM_functions.h"

const cv::Size FACE_IMGSIZE = cv::Size(50,50);
const int ELM_MODELS_COUNT = 4;
const int ELM_NHIDDENNODES = 32;

void refitEIEModel()
{
    ELM_IN_ELM_Model eieModel;
    
    int nModels = ELM_MODELS_COUNT;//超参1:elm模型数目
    eieModel.setInitPara(nModels,ELM_MODEL_PATH);
    //eieModel.loadStandardFaceDataset(FACEDB_PATH,1,FACE_IMGSIZE.width,FACE_IMGSIZE.height);//超参2:resize大小
    
    std::vector<std::string> label_strings;
    getSubDirs(FACEDB_PATH,label_strings);
    std::vector<std::string> names;
    cv::Mat feats;
    g_featEX.loadFeats(feats,names);
    eieModel.inputFaceFeats(label_strings,names,feats);
    
    for(int i=0;i<nModels;i++)
        eieModel.setSubModelHiddenNodes(i,ELM_NHIDDENNODES);//超参3:elm隐藏层节点数
    
    eieModel.fitSubModels_faceFeat();
    eieModel.fitMainModel_faceFeat();
    
    eieModel.save();
}

void updateFeatDb()
{
    std::string cmd_rm_featsFile = "rm " + FEATS_PATH;
    if(access(cmd_rm_featsFile.data(),F_OK) != -1)
        system(cmd_rm_featsFile.data());
    
    //std::map<std::string,bool> dbFiles;
    //if(access(HASH_FILE_PATH.data(),F_OK) != -1)
    //    dlib::deserialize(HASH_FILE_PATH) >> dbFiles;
    
    std::map<std::string,std::string> files;
    getFiles(FACEDB_PATH,files);
    
    std::vector<std::string> names;
    cv::Mat feats;
    
    for(std::map<std::string, std::string>::iterator it = files.begin();it != files.end();it++)
    {
        //文件不在库中，则提取特征
        //if(dbFiles.find(it->first) == dbFiles.end())
        //{
            cv::Mat src = cv::imread(it->first);
            if(src.empty())
                continue;
            
            cv::Mat feat;
            faceImgPreprocessing(src,feat);
            
            //g_featEX.saveFeat_add(it->second,feat);
            //dbFiles.insert(std::pair<std::string,bool>(it->first,true));
            
            names.push_back(it->second);
            feats.push_back(feat);
        //}
    }
    
    //dlib::serialize(HASH_FILE_PATH) << dbFiles;
    
    g_featEX.saveFeats_overwrite(names,feats);
}

void updateResnetDb()
{
    std::map<std::string, std::string> files;
    getFiles_less(FACEDB_PATH, files);
    
    std::vector<std::string> names;
    cv::Mat feats;
    int id = 0;
    for(std::map<std::string, std::string>::iterator it = files.begin();it != files.end();it++)
    {
        std::cout << "name:" << it->second << "	filepath:" <<it->first<<std::endl;
        
        cv::Mat frame = cv::imread(it->first);
        if(frame.empty())
            continue;
        
        //计算特征
        cv::Mat feat;
        g_featEX.resnetEx(frame,feat);
        
        if(feats.empty())
            feats.create(files.size(),feat.cols,CV_32F);
        
        names.push_back(it->second);
        feat.copyTo(feats.rowRange(id,id+1));
        id++;
        
        //test
        //std::cout<<"fileName:"<<it->second<<std::endl;
        //std::cout<<"resnet feat:\n"<<feat<<std::endl;
    }
    
    g_featEX.saveFeats_resnet(names,feats);
}

void handleFaceDb(int method)
{
    std::map<std::string, std::string> files;
    getFiles(FACEDB_PATH,files);
    
    if(files.empty())
    {
        std::cout<<"face database is empty"<<std::endl;
        return;
    }
    
    //人脸检测初始化
    FaceDetection detection;

    //对库中图像进行人脸检测并裁剪、对齐
    for(std::map<std::string, std::string>::iterator it = files.begin(); it != files.end(); it++)
    {
        cv::Mat image = cv::imread(it->first);
        
        if(isMarkedImg(image))
            continue;
        
        std::cout <<"handling file:" <<it->first<<std::endl;
        
        std::vector<cv::Rect> objects;
        detection.detect(image, objects);
        
        cv::Rect faceRect;
        if(objects.empty())
            faceRect = cv::Rect(0,0,image.cols-1,image.rows-1);
        else
            faceRect = objects[0];
        
        //std::cout<<"image.size: "<<image.size<<std::endl;
        //std::cout<<"faceRect: "<<faceRect<<std::endl;
        
        //对齐
        cv::Mat resultImg;
        g_featEX.alignFace(image,faceRect,resultImg);
        
        //cv::imshow("detect+alignment",resultImg);
        //cv::waitKey();
        
        //输出
        markImg(resultImg);//给处理过的图片打上标记，防止重复处理
        std::string outFile = it->first;
        outFile = outFile.substr(0,outFile.find_last_of("."));
        outFile += ".png";//jpg编码存取数据不一致，必须转成png格式
        remove(it->first.data());
        cv::imwrite(outFile,resultImg);
    }
    
    if(method == 1)
    {
        g_featEX.calcPCA();
        
        //更新特征库
        updateFeatDb();
        
        //重新训练elm-in-elm模型
        refitEIEModel();
    }
    
    if(method == 2)
    {
        //重新用resnet模型提取特征库
        updateResnetDb();
    }
}

void markImg(cv::Mat &img)
{
    int c = img.cols-1;
    int r = img.rows-1;
    
    img.at<cv::Vec3b>(0,0)[0] = 101;
    img.at<cv::Vec3b>(0,0)[1] = 100;
    img.at<cv::Vec3b>(0,0)[2] = 101;
    img.at<cv::Vec3b>(r,0)[0] = 100;
    img.at<cv::Vec3b>(r,0)[1] = 101;
    img.at<cv::Vec3b>(r,0)[2] = 100;
    img.at<cv::Vec3b>(0,c)[0] = 101;
    img.at<cv::Vec3b>(0,c)[1] = 100;
    img.at<cv::Vec3b>(0,c)[2] = 101;
    img.at<cv::Vec3b>(r,c)[0] = 100;
    img.at<cv::Vec3b>(r,c)[1] = 101;
    img.at<cv::Vec3b>(r,c)[2] = 100;
}

bool isMarkedImg(const cv::Mat &img)
{
    int c = img.cols-1;
    int r = img.rows-1;
    
    std::string key;
    key.append(std::to_string(img.at<cv::Vec3b>(0,0)[0]));
    key.append(std::to_string(img.at<cv::Vec3b>(0,0)[1]));
    key.append(std::to_string(img.at<cv::Vec3b>(0,0)[2]));
    key.append(std::to_string(img.at<cv::Vec3b>(r,0)[0]));
    key.append(std::to_string(img.at<cv::Vec3b>(r,0)[1]));
    key.append(std::to_string(img.at<cv::Vec3b>(r,0)[2]));
    key.append(std::to_string(img.at<cv::Vec3b>(0,c)[0]));
    key.append(std::to_string(img.at<cv::Vec3b>(0,c)[1]));
    key.append(std::to_string(img.at<cv::Vec3b>(0,c)[2]));
    key.append(std::to_string(img.at<cv::Vec3b>(r,c)[0]));
    key.append(std::to_string(img.at<cv::Vec3b>(r,c)[1]));
    key.append(std::to_string(img.at<cv::Vec3b>(r,c)[2]));
    
    //std::cout<<key<<std::endl;
    
    if(key == "101100101100101100101100101100101100")
        return true;
    else
        return false;
}

bool isEmptyFaceDb()
{
    std::map<std::string, std::string> files;
    getFiles(FACEDB_PATH,files);
    if(files.empty())
        return true;
    else
        return false;
}

void getSubDirs(std::string path, std::vector<std::string> &subDirs)
{
    DIR *dir;
	struct dirent *ptr;

	if(path[path.length()-1] != '/')
		path = path + "/";

	if((dir = opendir(path.c_str())) == nullptr)
	{
		std::cout<<"open the dir: "<< path <<" error!" <<std::endl;
		return;
	}
	
	while((ptr=readdir(dir)) !=nullptr )
	{
		///current dir OR parrent dir 
		if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0) 
            continue; 
		else if(ptr->d_type == 4)    ///dir
		{
            subDirs.push_back(ptr->d_name);
            std::cout<<ptr->d_name<<std::endl;
        }
	}
	
	closedir(dir);
}

void getFiles(std::string path, std::map<std::string, std::string> &files)
{
	DIR *dir;
	struct dirent *ptr;

	if(path[path.length()-1] != '/')
		path = path + "/";

	if((dir = opendir(path.c_str())) == nullptr)
	{
		std::cout<<"open the dir: "<< path <<" error!" <<std::endl;
		return;
	}
	
	while((ptr=readdir(dir)) !=nullptr )
	{
		///current dir OR parrent dir 
		if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0) 
            continue; 
		else if(ptr->d_type == 8) //file
		{
			std::string fn(ptr->d_name);
            
            std::string tail = fn.substr(fn.length()-3,fn.length()-1);
            if(tail == "dat" || tail == "xml")
                continue;
            
            std::string className = path;
            className.pop_back();
            className = className.substr(className.find_last_of("/")+1,className.length()-1);
            
            //if(fn.find(className) == std::string::npos)
            //    continue;
            
			std::string p = path + fn;
			files.insert(std::pair<std::string, std::string>(p, className));
		}
		else if(ptr->d_type == 10)    ///link file
		{}
		else if(ptr->d_type == 4)    ///dir
		{
            std::string p = path + std::string(ptr->d_name);
            getFiles(p,files);
        }
	}
	
	closedir(dir);
	return ;
}

void getFiles_less(std::string path, std::map<std::string, std::string> &files)
{
    DIR *dir;
	struct dirent *ptr;

	if(path[path.length()-1] != '/')
		path = path + "/";

	if((dir = opendir(path.c_str())) == nullptr)
	{
		std::cout<<"open the dir: "<< path <<" error!" <<std::endl;
		return;
	}
	
	while((ptr=readdir(dir)) !=nullptr )
	{
		///current dir OR parrent dir 
		if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0) 
            continue; 
		else if(ptr->d_type == 8) //file
        {
            std::string fn(ptr->d_name);
            
            std::string tail = fn.substr(fn.length()-3,fn.length()-1);
            if(tail == "dat" || tail == "xml")
                continue;
            
            std::string className = path;
            className.pop_back();
            className = className.substr(className.find_last_of("/")+1,className.length()-1);
            
            std::string p = path + fn;
            files.insert(std::pair<std::string, std::string>(p, className));
            
            break;
        }
        else if(ptr->d_type == 10)    ///link file
        {}
        else if(ptr->d_type == 4)    ///dir
        {
            std::string p = path + std::string(ptr->d_name);
            getFiles_less(p,files);
        }
	}
	
	closedir(dir);
	return ;
}

void getFileByName(std::string path, std::vector<cv::Mat> &imgs)
{
    DIR *dir;
	struct dirent *ptr;

	if(path[path.length()-1] != '/')
		path = path + "/";

	if((dir = opendir(path.c_str())) == nullptr)
	{
		std::cout<<"open the dir: "<< path <<" error!" <<std::endl;
		return;
	}
	
	while((ptr=readdir(dir)) !=nullptr )
	{
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0) 
            continue; 
		else if(ptr->d_type == 8) //file
		{
			std::string fn(ptr->d_name);
			std::string p = path + fn;
            
            imgs.push_back(cv::imread(p,0));
		}
    }
    
    closedir(dir);
}

void getAllFace(std::vector<cv::Mat> &faces)
{
    std::map<std::string, std::string> files;
    getFiles("./data/face_database",files);
    
    faces.resize(files.size());
    
    int traverseId=0;
    for(std::map<std::string, std::string>::iterator it = files.begin(); it != files.end(); it++)
    {
        cv::Mat image = cv::imread(it->first);
        
        cv::resize(image,image,cv::Size(50,50));
        
        faces[traverseId] = image.clone();
        traverseId++;
    }
}

void equalizeIntensity(cv::Mat &img)
{
    if(img.channels() == 3)
    {
        cv::Mat ycrcb;

        cv::cvtColor(img,ycrcb,cv::COLOR_BGR2YCrCb);

        std::vector<cv::Mat> channels;
        cv::split(ycrcb,channels);

        cv::equalizeHist(channels[0], channels[0]);

        cv::merge(channels,ycrcb);

        cv::cvtColor(ycrcb,img,cv::COLOR_YCrCb2BGR);
    }
    else
    {
        cv::equalizeHist(img,img);
    }
}

void saveMatAsXml(const cv::Mat &mat, std::string path)
{
    cv::FileStorage fswrite(path,cv::FileStorage::WRITE);
    
    fswrite<<"mat"<<mat;
    
    fswrite.release();
}

void loadXmlAsMat(std::string path, cv::Mat &mat)
{
    cv::FileStorage fsread(path,cv::FileStorage::READ);
    
    fsread["mat"]>>mat;
    
    fsread.release();
}

void faceImgPreprocessing(const cv::Mat &img, cv::Mat &feat)
{
    cv::Mat img2 = img.clone();
    cv::cvtColor(img2,img2,cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> cns;
    cv::split(img2,cns);
    cns[2].copyTo(img2);
    
    //缩放为标准尺寸
    cv::resize(img2,img2,FACE_IMGSIZE);
    
    //均衡化
    equalizeIntensity(img2);
    
    //特征提取
    g_featEX.extract(img2,feat);
    
    //归一化
    normalize(feat);
}

void faceImgPreprocessing(const std::vector<cv::Mat> &imgs, cv::Mat &feats)
{
    if(imgs.empty())
        return;
    
    cv::Mat feat;
    faceImgPreprocessing(imgs[0],feat);
    feats.create(imgs.size(),feat.cols,CV_32F);
    feat.copyTo(feats.rowRange(0,1));
    
    for(size_t i=1;i<imgs.size();i++)
    {
        faceImgPreprocessing(imgs[i],feat);
        feat.copyTo(feats.rowRange(i,i+1));
    }
}

void faceImgPreprocessing(const cv::Mat &img, cv::Mat &feat, std::string name)
{
    faceImgPreprocessing(img,feat);
    
    //输出
    g_featEX.saveFeat_add(name,feat);
}

void faceImgPreprocessing(const std::vector<cv::Mat> &imgs, cv::Mat &feats, std::vector<std::string> names)
{
    if(imgs.empty())
        return;
    
    cv::Mat feat;
    faceImgPreprocessing(imgs[0],feat);
    feats.create(imgs.size(),feat.cols,CV_32F);
    feat.copyTo(feats.rowRange(0,1));
    
    for(size_t i=1;i<imgs.size();i++)
    {
        faceImgPreprocessing(imgs[i],feat);
        feat.copyTo(feats.rowRange(i,i+1));
    }
    
    g_featEX.saveFeats_overwrite(names,feats);
}
