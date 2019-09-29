#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "SignUpDialog.h"
#include "params.h"
#include "FaceDetection/FaceDetection.h"
#include "FeatExtraction/FeatExtraction.h"
#include "FaceRecognition/FaceRecognition.h"
#include "functions.h"
#include <opencv2/highgui.hpp>
#include "debug.h"

const int NCANDIDATES = 1;
const float ELMREC_THRESHOLD = 0.90;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    
    //判断是否空的人脸库
    m_isEmptyRun = isEmptyFaceDb();
    
    //初始化：摄像头
    m_video = VIDEO_FILE;
    m_capture.open(m_video);
    
    if (!m_capture.isOpened())
    {
        std::cout << "Can't capture "<<m_video<< std::endl;
        exit(0);
    }
    
    //将timer与getframe连接
    connect(m_timer,SIGNAL(timeout()),this,SLOT(updateFrame()));
    m_timer->setInterval(1000/m_capture.get(cv::CAP_PROP_FPS));
    m_timer->start();
}

MainWindow::~MainWindow()
{
    delete ui;
    delete m_timer;
}

void MainWindow::showMat()
{
    cv::Mat frameToShow;
    cv::resize(m_frame,frameToShow,cv::Size(ui->label_Main->width()-2,ui->label_Main->height()-2));

    cv::cvtColor(frameToShow,frameToShow,cv::COLOR_BGR2RGB);
    m_qimgFrame = QImage((const uchar*)frameToShow.data,
                  frameToShow.cols,frameToShow.rows,
                  frameToShow.cols*frameToShow.channels(),
                  QImage::Format_RGB888);

    ui->label_Main->setPixmap(QPixmap::fromImage(m_qimgFrame));
}

void MainWindow::on_pushButton_SignUp_clicked()
{
    if(m_faceROI.empty())
        return;
    
    m_timer->stop();
    
    //创建登记窗口
    SignUpDialog *signUpDlg = new SignUpDialog();
    //登记窗口打开时停止其他窗口的运行
    signUpDlg->setWindowModality(Qt::ApplicationModal);
    //信息传递
    connect(signUpDlg, SIGNAL(sendData(std::string)), this, SLOT(addFace(std::string)));
    
    signUpDlg->setImg(m_faceROI);
    signUpDlg->show();
    signUpDlg->exec();
    
    delete signUpDlg;
    
    m_timer->start();
}

void orderNamesBySims(std::vector<std::string> &candidates, std::vector<float> &sims)
{
    for(size_t i=0;i<sims.size();i++)
    {
        for(size_t j=i+1;j<sims.size();j++)
        {
            if(sims[j] > sims[i])
            {
                float tmpSim = sims[i];
                sims[i] = sims[j];
                sims[j] = tmpSim;
                
                std::string tmpCandidate = candidates[i];
                candidates[i] = candidates[j];
                candidates[j] = tmpCandidate;
            }
        }
    }
}

/*
void MainWindow::showNames(std::vector<std::string> &candidates, std::vector<float> &sims)
{
    QString qstr;
    for(size_t i=0;i<sims.size();i++)
    {
        qstr.append(candidates[i].data());
        qstr.append(":");
        qstr.append(std::to_string(sims[i]).substr(0,6).data());
        qstr.append("\n");
    }
    ui->label_names->setText(qstr);
}
*/

void MainWindow::updateFrame()
{
    m_capture >> m_frameSrc;
    
    if(m_frameSrc.empty())
        exit(0);
    
    cv::flip(m_frameSrc,m_frameSrc,1);
    m_frameSrc.copyTo(m_frame);
    
    //进行检测
    std::vector<cv::Rect> objects;
    g_faceDT.detect(m_frame,objects);
    
    if(!objects.empty())
    {
        //人脸对齐
        //g_featEX.alignFace(m_frameSrc,objects[0],m_faceROI);
        bool sPos = g_featEX.judgeFaceAndAlign(m_frameSrc,objects[0],m_faceROI);
        
        //绘制检测结果
        cv::rectangle(m_frame,objects[0],cv::Scalar(0,255,255),2);
        
        //若人脸未正对摄像头则不识别并给出提示
        if(!sPos){
            cv::putText(m_frame,"Please keep your face to the camera",cv::Point(0,m_frame.rows),1,1.2,cv::Scalar(0,0,255));
        }
        
        if(!m_isEmptyRun && sPos)
        {
            std::string name;
             
            if(REC_METHOD == "resnet")
            {
                name = g_faceRC.recognize_resnetOnly(m_faceROI);
            }
            
            if(REC_METHOD == "elm")
            {
                int n = NCANDIDATES;
                std::vector<float> sims;
                std::vector<std::string> candidates;
                cv::Mat feat;
                faceImgPreprocessing(m_faceROI,feat);
                if(!feat.empty())
                {
                    g_faceRC.getCandidatesByELM(feat,n,candidates,sims);
                    
                    //name = g_faceRC.recognize_byFeat(feat,candidates,sims);
                    
                    //orderNamesBySims(candidates,sims);
                    
                    // /*
                    sims[0] = g_featEX.getMaxSim(feat,name);
                    if(sims[0] >= ELMREC_THRESHOLD)
                    {
                        //std::cout<<sims[0]<<std::endl;
                        //DBG("")
                        name = candidates[0];
                        cv::putText(m_frame,std::to_string(sims[0]).substr(0,5),cv::Point(objects[0].x+objects[0].width/2,objects[0].y),1,2,cv::Scalar(255,100,0),2);
                    }
                    else
                        name = "others";
                    // */
                    
                    //showNames(candidates,sims);
                    
                    //for(int i=0;i<n;i++)
                    //    std::cout<<"name:"<<candidates[i]<<" sim:"<<sims[i]<<std::endl;
                }
            }
            
            //绘制识别结果
            cv::putText(m_frame,name,objects[0].tl(),1,2,cv::Scalar(255,100,0),2);
        }
    }
    else
    {
        //ui->label_names->clear();
    }
    
    showMat();
}


void MainWindow::addFace(std::string name)
{
    std::string filename = FACEDB_PATH + "/" + name;
    
    //若不存在则新建
    bool isNewClass = 0;
    if(access(filename.data(),F_OK) == -1)
    {
        mkdir(filename.data(),00777);
        isNewClass = 1;
    }
    
    filename += "/";
    
    //输出
    time_t t = time(nullptr);
    char strTime[64];
    strftime(strTime, 64, "%Y-%m-%d-%H-%M-%S", localtime(&t));
    filename += name+std::string(strTime) + ".png";
    
    markImg(m_faceROI);
    cv::imwrite(filename,m_faceROI);
    if(m_isEmptyRun)
        m_isEmptyRun = false;
    
    //更新数据库
    if(REC_METHOD == "resnet" && isNewClass)
    {
        g_featEX.addResnetFeat(m_faceROI,name);
    }
    if(REC_METHOD == "elm")
    {
        g_featEX.calcPCA();
        updateFeatDb();
        refitEIEModel();
        g_faceRC.reloadEIE();
        
        /*
        if(isNewClass)//需改进
        {
            refitEIEModel();
            g_faceRC.reloadEIE();
        }
        else
        {
            g_faceRC.EIEtrainNewFace(m_faceROI,name);
            g_faceRC.reloadEIE();
        }
        */
    }
}
