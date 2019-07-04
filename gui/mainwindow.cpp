#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "SignUpDialog.h"
#include "params.h"
#include "FaceDetection/FaceDetection.h"
#include "FeatExtraction/FeatExtraction.h"
#include "FaceRecognition/FaceRecognition.h"
#include "functions.h"

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
    /*
    if(m_faceROI_src.empty())
        return;
    
    m_timer->stop();
    
    //创建登记窗口
    SignUpDialog *signUpDlg = new SignUpDialog();
    //登记窗口打开时停止其他窗口的运行
    signUpDlg->setWindowModality(Qt::ApplicationModal);
    //信息传递
    connect(signUpDlg, SIGNAL(sendData(bool, std::string)), this, SLOT(addFace(bool, std::string)));
    
    signUpDlg->setImg(m_faceROI_src);
    signUpDlg->show();
    signUpDlg->exec();
    
    delete signUpDlg;
    
    m_timer->start();
    */
}

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
        g_featEX.alignFace(m_frameSrc,objects[0],m_faceROI);
        
        //绘制检测结果
        cv::rectangle(m_frame,objects[0],cv::Scalar(0,255,255),2);
        
        if(!m_isEmptyRun)
        {
            std::string name;
             
            if(REC_METHOD == "resnet")
            {
                name = g_faceRC.recognize_resnetOnly(m_faceROI);
            }
            
            /*
            if(m_rec.method == "elm")
            {
                int n = 5;
                std::map<float,std::string> nameScores;
                isInFaceDb = m_rec.recognize(m_faceROI,n,nameScores);
                
                showNames(nameScores);
                
                //for(int i=0;i<n;i++)
                //    std::cout<<"names["<<i<<"]:"<<names[i]<<std::endl;
                
                if(!nameScores.empty())
                    name = nameScores.rbegin()->second;
                else
                    name = "others";
            }*/
            
            //显示识别结果
            cv::putText(m_frame,name,objects[0].tl(),1,2,cv::Scalar(255,100,0),2);
        }
    }
    else
    {
        //ui->label_names->clear();
    }
    
    showMat();
}


void MainWindow::addFace(bool isSignUp, std::string name)
{
    /*
    if(!isSignUp || name.empty())
        return;
        
    std::string filename = "./data/face_database/" + name;
    
    //若不存在则新建
    bool isNewClass = 0;
    if(access(filename.data(),F_OK) == -1)
    {
        mkdir(filename.data(),00777);
        isNewClass = 1;
    }
    
    filename += "/";
    
    //输出
    if(isNewClass)
    {
        filename += name + ".png";
    }
    else
    {
        time_t t = time(0);
        char strTime[64];
        strftime(strTime, 64, "%Y-%m-%d-%H-%M-%S", localtime(&t));
        
        filename += name+std::string(strTime) + ".png";
    }
    
    markImg(m_faceROI_src);
    cv::imwrite(filename,m_faceROI_src);
    if(isEmptyRun)
    {
        filename = "./data/face_database/" + name + "/" + name + "2.png";
        cv::imwrite(filename,m_faceROI_src);
        isEmptyRun = false;
    }
    
    //更新数据库
    if(m_rec.method == "resnet" && isNewClass)
    {
        m_rec.init_updateResnetDb();
    }
    if(m_rec.method == "elm")
    {
        if(isNewClass)
            m_rec.init_updateEIEdb();
        else
            m_rec.updateEIEdb(m_faceROI,name);
    }
    */
}
