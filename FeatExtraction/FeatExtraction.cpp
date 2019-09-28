#include "FeatExtraction.h"
#include "params.h"
#include <dlib/opencv/cv_image.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/features2d.hpp>
//#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include "functions.h"

const std::string FEATEX_METHOD = "pca";

const std::string PCA_PATH = "./data/face_database/pca.xml";

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
    if(m_method == "pca")
    {
        pcaEx(img,feat);
        return;
    }
    
    if(m_method == "resnet")
    {
        resnetEx(img,feat);
        return;
    }
    
    if(m_method == "highDimLbp")
    {
        highDimLbpEx(img,feat);
        return;
    }
    
    if(m_method == "multi")
    {
        multiFeatEx(img,feat);
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

void FeatExtraction::highDimLbpEx(const cv::Mat &faceMat, cv::Mat &feat)
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
    for(size_t i=0;i<tmpFeat.size();i++)
        feat.at<float>(0,i) = tmpFeat[i];
}

cv::Mat asRowMatrix(const std::vector<cv::Mat>& src, int rtype)
{
    //std::cout<<"asRowMatrix input 0:"<<src[0]<<std::endl;
    
    //样本数量
    size_t n = src.size();
    //如果没有样本，返回空矩阵
    if(n == 0)
        return cv::Mat();
    //样本的维数
    size_t d = src[0].total()*src[0].channels();
    //std::cout<<"d:"<<d<<std::endl;

    cv::Mat data(n, d, rtype);
    //拷贝数据
    for(int i = 0; i < n; i++)
    {
        cv::Mat xi = data.row(i);
        //转化为1行，n列的格式
        if(src[i].isContinuous())
        {
            src[i].reshape(1, 1).convertTo(xi, rtype);
        } 
        else{
            src[i].clone().reshape(1, 1).convertTo(xi, rtype);
        }
    }
    
    //std::cout<<"asRowMatrix output:"<<data<<std::endl;
    
    return data;
}

void FeatExtraction::pcaEx(const cv::Mat &faceMat, cv::Mat &feat)
{
    if(m_pca.eigenvectors.empty())
        loadPCA();
    
    std::vector<cv::Mat> tmpFaces;
    tmpFaces.push_back(faceMat);
    
    cv::Mat data = asRowMatrix(tmpFaces,CV_32F);
    
    //std::cout<<"m_PCA_mean:"<<m_pca.mean<<std::endl;
    //std::cout<<"m_PCA_eigenVecs:"<<m_pca.eigenvectors<<std::endl;
    //std::cout<<"data:"<<data<<std::endl;
    
    //cv::PCAProject(data,m_PCA_mean,m_PCA_eigenVecs,feat);
    m_pca.project(data,feat);
    
    //std::cout<<"feat:"<<feat<<std::endl;
}

void FeatExtraction::calcPCA()
{
    //载入数据
    std::vector<cv::Mat> faces;
    getAllFace(faces);
    
    cv::Mat data = asRowMatrix(faces,CV_32F);
    
    //cv::Mat averageFace = cv::imread("./data/models/averageFace.png");
    //std::vector<cv::Mat> tmpFaces;
    //tmpFaces.push_back(averageFace);
    //cv::Mat averageData = asRowMatrix(tmpFaces,CV_32F);
    
    //m_pca = cv::PCA(data, averageData, cv::PCA::DATA_AS_ROW, 0.99);
    m_pca = cv::PCA(data, cv::Mat(), cv::PCA::DATA_AS_ROW, 0.99);
    
    //m_PCA_mean = averageData;
    //m_PCA_eigenVecs = pca.eigenvectors;
    
    //计算DTD
    //cv::Mat D = data.clone();
    //for(int i=0;i<data.rows;i++)
    //    data.row(i) -= averageData;
    
    //m_PCA_DTD = D.t() * D;
    
    //std::cout<<"m_PCA_mean:"<<m_pca.mean<<std::endl;
    //std::cout<<"m_PCA_eigenVecs:"<<m_pca.eigenvectors<<std::endl;
    //std::cout<<"data:"<<data<<std::endl;
    
    //保存
    savePCA();
}

void FeatExtraction::updatePCA(const cv::Mat &newFace)
{
    if(m_PCA_DTD.empty())
        loadPCA();
    
    
}

void FeatExtraction::loadPCA()
{
    /*
    cv::FileStorage fsread(PCA_PATH,cv::FileStorage::READ);
    fsread["mean"]>>m_PCA_mean;
    fsread["eigenVecs"]>>m_PCA_eigenVecs;
    fsread["DTD"]>>m_PCA_DTD;
    fsread.release();
    */
    
    cv::FileStorage fsread(PCA_PATH,cv::FileStorage::READ);
    m_pca.read(fsread.root());
}

void FeatExtraction::savePCA()
{
    /*
    cv::FileStorage fswrite(PCA_PATH,cv::FileStorage::WRITE);
    fswrite<<"mean"<<m_PCA_mean;
    fswrite<<"eigenVecs"<<m_PCA_eigenVecs;
    fswrite<<"DTD"<<m_PCA_DTD;
    fswrite.release();
    */
    
    cv::FileStorage fswrite(PCA_PATH,cv::FileStorage::WRITE);
    m_pca.write(fswrite);
}

void FeatExtraction::multiFeatEx(const cv::Mat &faceMat, cv::Mat &feat)
{
    /*
    //得到68个landmarks
    cv::Rect rect(0,0,faceMat.cols-1,faceMat.rows-1);
    dlib::full_object_detection shape;
    getShape(faceMat,rect,shape);
    std::vector<cv::Point2f> landmarks;
    dlibPoint2cvPoint2f(shape,landmarks);
    
    //填充周围
    float paddingRatio = 0.2;
    cv::Mat faceMat_pad = padImg(faceMat,paddingRatio);
    int ew = faceMat.cols*paddingRatio;
    int eh = faceMat.rows*paddingRatio;
    for(size_t i=0;i<landmarks.size();i++)
    {
        landmarks[i].x += ew;
        landmarks[i].y += eh;
    }
    
    //point2f转keypoint
    std::vector<cv::KeyPoint> kp;
    cv::KeyPoint::convert(landmarks,kp);
    
    
    //test
    cv::Mat testImg = faceMat_pad.clone();
    for(size_t i=0;i<landmarks.size();i++)
        cv::circle(testImg,cv::Point(landmarks[i].x,landmarks[i].y),2,cv::Scalar(0,0,255));
    cv::imshow("faceMat_pad",testImg);
    cv::waitKey();
    
    
    //计算特征描述子
    std::vector<cv::Mat> descs;
    
    //double st = cv::getTickCount();
    
    cv::Mat tmpDesc;
    cv::Ptr<cv::DescriptorExtractor> orbEx = cv::ORB::create();
    orbEx->compute(faceMat_pad, kp, tmpDesc);
    descs.push_back(tmpDesc);
    
    //double rt = (cv::getTickCount()-st)/cv::getTickFrequency();
    //std::cout<<"0:"<<rt<<std::endl;
    //st = cv::getTickCount();
    
    tmpDesc = cv::Mat();
    cv::Ptr<cv::DescriptorExtractor> freakEx = cv::xfeatures2d::FREAK::create();
    freakEx->compute(faceMat_pad, kp, tmpDesc);
    descs.push_back(tmpDesc);
    
    //rt = (cv::getTickCount()-st)/cv::getTickFrequency();
    //std::cout<<"2:"<<rt<<std::endl;
    //st = cv::getTickCount();
    
    tmpDesc = cv::Mat();
    cv::Ptr<cv::DescriptorExtractor> siftEx = cv::xfeatures2d::SIFT::create();
    siftEx->compute(faceMat_pad, kp, tmpDesc);
    descs.push_back(tmpDesc);
    
    //rt = (cv::getTickCount()-st)/cv::getTickFrequency();
    //std::cout<<"3:"<<rt<<std::endl;
    //st = cv::getTickCount();
    
    tmpDesc = cv::Mat();
    cv::Ptr<cv::DescriptorExtractor> surfEx = cv::xfeatures2d::SURF::create();
    surfEx->compute(faceMat_pad, kp, tmpDesc);
    descs.push_back(tmpDesc);
    
    //rt = (cv::getTickCount()-st)/cv::getTickFrequency();
    //std::cout<<"4:"<<rt<<std::endl;
    //st = cv::getTickCount();
    
    tmpDesc = cv::Mat();
    cv::Ptr<cv::DescriptorExtractor> latchEx = cv::xfeatures2d::LATCH::create();
    latchEx->compute(faceMat_pad, kp, tmpDesc);
    descs.push_back(tmpDesc);
    
    //rt = (cv::getTickCount()-st)/cv::getTickFrequency();
    //std::cout<<"6:"<<rt<<std::endl;
    //st = cv::getTickCount();
    
    tmpDesc = cv::Mat();
    cv::Ptr<cv::DescriptorExtractor> vggEx = cv::xfeatures2d::VGG::create();
    vggEx->compute(faceMat_pad, kp, tmpDesc);
    descs.push_back(tmpDesc);
    
    //rt = (cv::getTickCount()-st)/cv::getTickFrequency();
    //std::cout<<"7:"<<rt<<std::endl;
    //st = cv::getTickCount();
    
    tmpDesc = cv::Mat();
    cv::Ptr<cv::DescriptorExtractor> lucidEx = cv::xfeatures2d::LUCID::create();
    lucidEx->compute(faceMat_pad, kp, tmpDesc);
    descs.push_back(tmpDesc);
    
    //rt = (cv::getTickCount()-st)/cv::getTickFrequency();
    //std::cout<<"8:"<<rt<<std::endl;
    //st = cv::getTickCount();
    
    tmpDesc = cv::Mat();
    cv::Ptr<cv::DescriptorExtractor> boostEx = cv::xfeatures2d::BoostDesc::create();
    boostEx->compute(faceMat_pad, kp, tmpDesc);
    descs.push_back(tmpDesc);
    
    //rt = (cv::getTickCount()-st)/cv::getTickFrequency();
    //std::cout<<"9:"<<rt<<std::endl;
    
    //descs转为feat
    int featLength = 0;
    for(size_t i=0;i<descs.size();i++)
    {
        //std::cout<<"id:"<<i<<std::endl;
        //std::cout<<descs[i].size<<std::endl;
        //std::cout<<descs[i]<<std::endl;
        
        if(descs[i].rows < 68)
            return;
        
        featLength += descs[i].cols;
        
        descs[i].convertTo(descs[i],CV_32F);
        cv::normalize(descs[i],descs[i],1,0,cv::NORM_INF);
        
        //std::cout<<descs[i]<<std::endl;
    }
    featLength *= 68;
    
    //std::cout<<"featLength:"<<featLength<<std::endl;
    
    feat.create(1,featLength,CV_32F);
    int idx = 0;
    for(size_t i=0;i<descs.size();i++)
    {
        for(int x=0;x<descs[i].cols;x++)
            for(int y=0;y<descs[i].rows;y++)
            {
                feat.at<float>(0,idx) = descs[i].at<float>(y,x);
                idx++;
            }
    }
    */
    
    /*
    //int blockSize = 9;
    for(int i=0;i<landmarks.size();i++)
    {
        //得到landmarks[i]的邻域
        cv::Mat nbBlock;
        getNbBlock(faceMat,landmarks[i],blockSize,nbBlock);
        
        //提取特征描述子
    }
    */
}

cv::Mat FeatExtraction::padImg(const cv::Mat &img, float paddingRatio)
{
    int w = img.cols;
    int h = img.rows;
    
    int ew = w*paddingRatio;
    int eh = h*paddingRatio;
    
    w += 2*ew;
    h += 2*eh;
    
    cv::Mat img2(h,w,img.type());
    if(img2.channels()==3)
        img2 = cv::Scalar(0,0,0);
    else
        img2 = cv::Scalar(0);
    
    cv::Mat ROI = img2(cv::Range(eh,eh+img.rows),cv::Range(ew,ew+img.cols));
    img.copyTo(ROI);
    
    return img2;
}

/*
void FeatExtraction::getNbBlock(const cv::Mat &img, cv::Point center, int blockSize, cv::Mat &block)
{
    int halfLen = blockSize/2;
    
    bool left = false;
    bool right = false;
    bool top = false;
    bool bottom = false;
    
    int x1 = center.x - halfLen;
    if(x1 < 0) {x1 = 0; left = true;}

    int y1 = center.y - halfLen;
    if(y1 < 0) {y1 = 0; top = true;}
    
    int x2 = center.x + halfLen;
    if(x2 > img.cols-1) {x2 = img.cols-1; right = true;}
    
    int y2 = center.y + halfLen;
    if(y2 > img.rows-1) {y2 = img.rows-1; bottom = true;}
    
    cv::Mat tmpBlock = img(cv::Rect(x1,y1,x2-x1,y2-y1));
    
    block.create(blockSize,blockSize,img.type());
    if(block.channels()==3)
        block = cv::Scalar(0,0,0);
    else
        block = cv::Scalar(0);
    
    if(left && top)
    {
        cv::Mat ROI = block(cv::Range(0,tmpBlock.rows),cv::Range(0,tmpBlock.cols));
        tmpBlock.copyTo(ROI);
    }
    else if(left && bottom)
    {
        cv::Mat ROI = block(cv::Range(0,tmpBlock.rows),cv::Range(0,tmpBlock.cols));
        tmpBlock.copyTo(ROI);
    }
    else if(right && top)
    {
        
    }
    else if(right && bottom)
    {
        
    }
}
*/

void FeatExtraction::dlibPoint2cvPoint(const dlib::full_object_detection &S, std::vector<cv::Point> &L)
{
    for(unsigned int i = 0; i<S.num_parts();++i)
        L.push_back(cv::Point(S.part(i).x(),S.part(i).y()));
}

void FeatExtraction::dlibPoint2cvPoint2f(const dlib::full_object_detection &S, std::vector<cv::Point2f> &L)
{
    for(unsigned int i = 0; i<S.num_parts();++i)
        L.push_back(cv::Point2f(S.part(i).x(),S.part(i).y()));
}

void FeatExtraction::cvRect2dlibRect(const cv::Rect &cvRec, dlib::rectangle &dlibRec)
{
    dlibRec = dlib::rectangle((long)cvRec.tl().x, (long)cvRec.tl().y, (long)cvRec.br().x - 1, (long)cvRec.br().y - 1);
}

void FeatExtraction::drawShape(cv::Mat &img, dlib::full_object_detection shape)
{
    //shape转化为landmarks
    std::vector<cv::Point> landmarks;
    dlibPoint2cvPoint(shape,landmarks);
    
    //test:在两眼之间画线
    for(size_t i=0;i<landmarks.size();i++)
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
    int x1 = rect.x;
    int y1 = rect.y;
    int x2 = rect.br().x;
    int y2 = rect.br().y;
    
    if(x1 < 0)
        x1 = 0;
    if(y1 < 0)
        y1 = 0;
    if(x2 > imgSize.width-1)
        x2 = imgSize.width-1;
    if(y2 > imgSize.height-1)
        y2 = imgSize.height-1;
    
    rect = cv::Rect(cv::Point(x1,y1),cv::Point(x2,y2));
}

void modifyRectByFacePt(const dlib::full_object_detection &shape, cv::Rect &rect)
{
    std::vector<cv::Point> landmarks;
    for(unsigned int i = 0; i<shape.num_parts();++i)
        landmarks.push_back(cv::Point(shape.part(i).x(),shape.part(i).y()));
    
    int l = landmarks[0].x;
    int r = landmarks[0].x;
    int t = landmarks[0].y;
    int b = landmarks[0].y;
    
    for(size_t i=0;i<landmarks.size();i++)
    {
        if(landmarks[i].x < l)
            l = landmarks[i].x;
        
        if(landmarks[i].x > r)
            r = landmarks[i].x;
        
        if(landmarks[i].y < t)
            t = landmarks[i].y;
        
        if(landmarks[i].y > b)
            b = landmarks[i].y;
    }
    
    float expandRatio = 0.05;
    
    int ew = (r-l)*expandRatio;
    int eh = (b-t)*expandRatio;
    
    l -= ew;
    r += ew;
    t -= eh;
    b += eh;
    
    rect = cv::Rect(l,t,r-l,b-t);
    
    /*
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
    */
}

void FeatExtraction::alignFace(const cv::Mat &inputImg, cv::Rect &faceRect, cv::Mat &resultImg)
{
    //获取特征点
    dlib::full_object_detection shape;
    getShape(inputImg,faceRect,shape);
    
    //shape转化为landmarks
    std::vector<cv::Point> landmarks;
    dlibPoint2cvPoint(shape,landmarks);
    
    /*
    //test
    cv::Mat testShapeImg = inputImg.clone();
    drawShape(testShapeImg,shape);
    cv::rectangle(testShapeImg,faceRect,cv::Scalar(255,0,0),2);
    cv::namedWindow("testShapeImg0",0);
    cv::imshow("testShapeImg0",testShapeImg);
    */
    
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
    
    /*
    //test
    cv::Mat testShapeImg2 = rotatedImg.clone();
    drawShape(testShapeImg2,shape2);
    cv::rectangle(testShapeImg2,faceRect,cv::Scalar(255,0,0),2);
    cv::namedWindow("testShapeImg",0);
    cv::imshow("testShapeImg",testShapeImg2);
    */
    
    resultImg = rotatedImg(faceRect).clone();
}

void addLine(cv::Mat &mat, const cv::Mat &line)
{
    mat.push_back(line);
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

void FeatExtraction::loadFeats(cv::Mat &feats, std::vector<std::string> &names)
{
    cv::FileStorage fsread(FEATS_PATH,cv::FileStorage::READ);
    fsread["names"]>>names;
    fsread["feats"]>>feats;
    fsread.release();
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
    for(size_t i=0;i<tmpNames.size();i++)
    {
        std::string name = tmpNames[i];
        
        for(size_t j=0;j<candidates.size();j++)
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
    for(size_t i=0;i<featList.size();i++)
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

float FeatExtraction::getMaxSim(const cv::Mat &feat, std::string name)
{
    std::vector<std::string> candidates;
    candidates.push_back(name);
    
    cv::Mat feats;
    std::vector<std::string> names;
    loadFeats(feats,names);
    
    float maxSim = 0;
    for(int i=0;i<feats.rows;i++)
    {
        cv::Mat featInDB = feats.rowRange(i,i+1);
        float a = cv::norm(feat);
        float b = cv::norm(featInDB);
        float c = cv::norm(feat-featInDB);
        
        float sim = (a*a + b*b - c*c)/(2*a*b);
        
        if(sim > maxSim)
            maxSim = sim;
    }
    
    return maxSim;
}
