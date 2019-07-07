#include "FeatExtraction.h"
#include "params.h"
#include <dlib/opencv/cv_image.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

FeatExtraction g_featEX;

FeatExtraction::FeatExtraction()
{
    m_method.assign(FEATEX_METHOD);
    
    dlib::deserialize(SHAPE_PREDICTOR_PATH) >> m_shapePredictor;
    
    if(m_method == "resnet")
        dlib::deserialize(RESNET_MODEL_PATH) >> m_resnet;
}

void FeatExtraction::extract(const cv::Mat &img, cv::Mat &feat)
{
    if(m_method == "resnet")
    {
        resnetEx(img,feat);
        return;
    }
    
    if(m_method == "lbp")
    {
        lbpEx(img,feat);
        return;
    }
}

void FeatExtraction::extract_batch(const std::vector<cv::Mat> &imgs, cv::Mat &feats)
{
    if(imgs.empty())
        return;
    
    //利用第一张图片确定特征向量长度
    cv::Mat tmpFeat;
    extract(imgs[0],tmpFeat);
    int featLen = tmpFeat.cols;
    
    //确定feats尺寸
    feats.create(imgs.size(),featLen,CV_32F);
    
    //给第一个特征向量赋值
    tmpFeat.copyTo(feats.rowRange(0,1));
    
    //开始遍历、提取特征
    for(size_t i=1;i<imgs.size();i++)
    {
        cv::Mat row = feats.rowRange(i,i+1);
        extract(imgs[i],row);
    }
}

void FeatExtraction::resnetEx(const cv::Mat &faceMat, cv::Mat &feat)
{
    //转换图像格式
    cv::Mat faceMat2;
    if(faceMat.channels()==3)
        cv::cvtColor(faceMat, faceMat2, cv::COLOR_BGR2GRAY);
    else
        faceMat2 = faceMat;
    dlib::array2d<dlib::bgr_pixel> dimg;
    dlib::assign_image(dimg, dlib::cv_image<uchar>(faceMat2));
    
    //得到shape
    cv::Rect rect(0,0,faceMat.cols-1,faceMat.rows-1);
    dlib::full_object_detection shape;
    getShape(faceMat,rect,shape);
    
    //提取特征
    dlib::matrix<dlib::rgb_pixel> face_chip;
    dlib::extract_image_chip(dimg, dlib::get_face_chip_details(shape,150,0.25), face_chip);
    dlib::matrix<float,0,1> faceDescriptor = m_resnet(face_chip);
    
    //转化为Mat
    int rows = faceDescriptor.nr();
    int cols = faceDescriptor.nc();
    feat.create(1,rows*cols,CV_32F);
    int idx=0;
    for(int r=0;r<rows;r++)
    {
        for(int c=0;c<cols;c++)
        {
            feat.at<float>(0,idx) = faceDescriptor(r,c);
            idx++;
        }
    }
}

void FeatExtraction::lbpEx(const cv::Mat &faceMat, cv::Mat &feat)
{
    //转换图像格式
    cv::Mat faceMat2;
    if(faceMat.channels() == 1)
        cv::cvtColor(faceMat,faceMat2,cv::COLOR_GRAY2BGR);
    else
        faceMat2 = faceMat;
    dlib::array2d<unsigned char> c_gray_face;
    dlib::assign_image(c_gray_face,dlib::cv_image<dlib::bgr_pixel>(faceMat2));
    
    //得到shape
    cv::Rect rect(0,0,faceMat2.cols-1,faceMat2.rows-1);
    dlib::full_object_detection shape;
    getShape(faceMat2,rect,shape);
    
    //提取特征
    std::vector<float> tmpFeat;
    dlib::extract_highdim_face_lbp_descriptors(c_gray_face, shape, tmpFeat);
    
    //转换为Mat
    feat.create(1,tmpFeat.size(),CV_32F);
    for(int i=0;i<tmpFeat.size();i++)
        feat.at<float>(0,i) = tmpFeat[i];
}

void dlibPoint2cvPoint(const dlib::full_object_detection &S, std::vector<cv::Point> &L)
{
    for(unsigned int i = 0; i<S.num_parts();++i)
        L.push_back(cv::Point(S.part(i).x(),S.part(i).y()));
}

void cvRect2dlibRect(const cv::Rect &cvRec, dlib::rectangle &dlibRec)
{
    dlibRec = dlib::rectangle((long)cvRec.tl().x, (long)cvRec.tl().y, (long)cvRec.br().x - 1, (long)cvRec.br().y - 1);
}

void drawShape(cv::Mat &img, dlib::full_object_detection shape)
{
    //shape转化为landmarks
    std::vector<cv::Point> landmarks;
    dlibPoint2cvPoint(shape,landmarks);
    
    //test:在两眼之间画线
    for(int i=0;i<landmarks.size();i++)
    {
        cv::circle(img,landmarks[i],3,cv::Scalar(0,255,0),-1);
        cv::putText(img,std::to_string(i),landmarks[i],1,0.8,cv::Scalar(255,0,0));
    }
}

void FeatExtraction::getShape(const cv::Mat &inputImg, const cv::Rect &faceRect, dlib::full_object_detection &shape)
{
    cv::Mat img2 = inputImg.clone();
    if(inputImg.channels()==1)
        cv::cvtColor(img2,img2,cv::COLOR_GRAY2BGR);
    
    //转换opencv图像为dlib图像
    dlib::cv_image<dlib::bgr_pixel> cimg(img2);

    //提取脸部特征点(68个),存储在shape
    dlib::rectangle face_dlibRect;
    cvRect2dlibRect(faceRect,face_dlibRect);
    shape = m_shapePredictor(cimg,face_dlibRect);
    
    /*
    cv::Mat testImg = inputImg.clone();
    drawShape(testImg,shape);
    cv::imshow("test",testImg);
    cv::waitKey();
    */
    
}

void modifyROI(const cv::Size imgSize, cv::Rect &rect)
{
    if(rect.x < 0)
        rect.x = 0;
    if(rect.y < 0)
        rect.y = 0;
    if(rect.br().x > imgSize.width-1)
        rect.width = imgSize.width-1-rect.x;
    if(rect.br().y > imgSize.height-1)
        rect.height = imgSize.height-1-rect.y;
}

void modifyRectByFacePt(const dlib::full_object_detection &shape, cv::Rect &rect)
{
    std::vector<cv::Point> landmarks;
    for(unsigned int i = 0; i<shape.num_parts();++i)
        landmarks.push_back(cv::Point(shape.part(i).x(),shape.part(i).y()));
    
    int b = landmarks[8].y;
    
    int l1 = landmarks[0].x;
    int l2 = landmarks[1].x;
    int l3 = landmarks[17].x;
    int l = l1<l2?(l1<l3?l1:l3):(l2<l3?l2:l3);
    
    int r1 = landmarks[26].x;
    int r2 = landmarks[15].x;
    int r3 = landmarks[16].x;
    int r = r1>r2?(r1>r3?r1:r3):(r2>r3?r2:r3);
    
    int t1 = landmarks[19].y;
    int t2 = landmarks[24].y;
    int t = t1<t2?t1:t2;
    
    int refLen = (r-l)/20;
    
    rect.x = l-refLen;
    rect.y = t-2*refLen;
    rect.width = r-rect.x+refLen;
    rect.height = b-rect.y;
}

void FeatExtraction::alignFace(const cv::Mat &inputImg, cv::Rect &faceRect, cv::Mat &resultImg)
{
    //获取特征点
    dlib::full_object_detection shape;
    getShape(inputImg,faceRect,shape);
    
    //shape转化为landmarks
    std::vector<cv::Point> landmarks;
    dlibPoint2cvPoint(shape,landmarks);
    
    //以两眼连线偏角绕矩形框中心旋转整幅图像
    cv::Point leye = landmarks[36];
    cv::Point reye = landmarks[45];
    double dy = reye.y - leye.y; 
    double dx = reye.x - leye.x; 
    double angle = atan2(dy, dx) * 180.0 / CV_PI; 
    cv::Point center = cv::Point(faceRect.x+faceRect.width/2,faceRect.y+faceRect.height/2);
    cv::Mat rotMat = cv::getRotationMatrix2D(center, angle, 1); 
    cv::Mat rotatedImg;
    cv::warpAffine(inputImg, rotatedImg, rotMat, inputImg.size());
    
    //取这时矩形框内的部分作为人脸
    dlib::full_object_detection shape2;
    getShape(rotatedImg,faceRect,shape2);
    modifyRectByFacePt(shape2,faceRect);
    modifyROI(rotatedImg.size(),faceRect);
    
    resultImg = rotatedImg(faceRect).clone();
}

void addLine(cv::Mat &mat, const cv::Mat &line)
{
    int n = mat.rows+1;
    cv::Mat newMat(n,mat.cols,CV_32F);
    mat.copyTo(newMat.rowRange(0,n-1));
    line.copyTo(newMat.rowRange(n-1,n));
    newMat.copyTo(mat);
}

void FeatExtraction::saveFeat_add(std::string name, const cv::Mat &feat)
{
    std::vector<std::string> names;
    cv::Mat feats;
    
    cv::FileStorage fsread(FEATS_PATH,cv::FileStorage::READ);
    fsread["names"]>>names;
    fsread["feats"]>>feats;
    fsread.release();
    
    names.push_back(name);
    addLine(feats,feat);
    
    cv::FileStorage fswrite(FEATS_PATH,cv::FileStorage::WRITE);
    fswrite<<"names"<<names;
    fswrite<<"feats"<<feats;
    fswrite.release();
}

void FeatExtraction::saveFeats_overwrite(std::vector<std::string> names, const cv::Mat &feats)
{
    cv::FileStorage fswrite(FEATS_PATH,cv::FileStorage::WRITE);
    fswrite<<"names"<<names;
    fswrite<<"feats"<<feats;
    fswrite.release();
}

void FeatExtraction::loadFeats(const std::vector<std::string> &candidates, 
                               cv::Mat &feats, std::vector<std::string> &names)
{
    cv::Mat tmpFeats;
    std::vector<std::string> tmpNames;
    
    cv::FileStorage fsread(FEATS_PATH,cv::FileStorage::READ);
    fsread["names"]>>tmpNames;
    fsread["feats"]>>tmpFeats;
    fsread.release();
    
    std::vector<cv::Mat> featList;
    for(int i=0;i<tmpNames.size();i++)
    {
        std::string name = tmpNames[i];
        
        for(int j=0;j<candidates.size();j++)
        {
            if(name == candidates[j])
            {
                cv::Mat feat = tmpFeats.rowRange(i,i+1);
                names.push_back(name);
                
                featList.push_back(feat);
                break;
            }
        }
    }
    
    if(featList.empty())
        return;
    
    feats.create(featList.size(),featList[0].cols,CV_32F);
    for(int i=0;i<featList.size();i++)
    {
        cv::Mat ROI = feats.rowRange(i,i+1);
        featList[i].copyTo(ROI);
    }
}

void FeatExtraction::saveFeats_resnet(std::vector<std::string> names, const cv::Mat &feats)
{
    cv::FileStorage fswrite(RESNET_FEATS_PATH,cv::FileStorage::WRITE);
    fswrite<<"names"<<names;
    fswrite<<"feats"<<feats;
    fswrite.release();
}

void FeatExtraction::loadFeats_resnet(std::vector<std::string> &names, cv::Mat &feats)
{
    cv::FileStorage fsread(RESNET_FEATS_PATH,cv::FileStorage::READ);
    fsread["names"]>>names;
    fsread["feats"]>>feats;
    fsread.release();
}

void FeatExtraction::addResnetFeat(const cv::Mat &faceImg, const std::string &name)
{
    cv::Mat feat;
    resnetEx(faceImg,feat);
    
    std::vector<std::string> names;
    cv::Mat feats;
    
    cv::FileStorage fsread(RESNET_FEATS_PATH,cv::FileStorage::READ);
    fsread["names"]>>names;
    fsread["feats"]>>feats;
    fsread.release();
    
    names.push_back(name);
    addLine(feats,feat);
    
    cv::FileStorage fswrite(RESNET_FEATS_PATH,cv::FileStorage::WRITE);
    fswrite<<"names"<<names;
    fswrite<<"feats"<<feats;
    fswrite.release();
}
