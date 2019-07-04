#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <opencv2/videoio.hpp>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    
private:
    Ui::MainWindow *ui;
    
    bool m_isEmptyRun;
    
    std::string m_video;
    cv::VideoCapture m_capture;
    
    cv::Mat m_frame;
    cv::Mat m_frameSrc;
    
    cv::Mat m_faceROI;
    
    QImage m_qimgFrame;
    QTimer *m_timer = new QTimer(this);
    
    void showMat();
    
private slots:
    void updateFrame();
    void on_pushButton_SignUp_clicked();
    void addFace(bool isSignUp, std::string name);
};

#endif // MAINWINDOW_H
