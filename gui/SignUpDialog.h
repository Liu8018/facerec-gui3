#ifndef SIGNUPDIALOG_H
#define SIGNUPDIALOG_H

#include <QDialog>
#include <opencv2/imgproc.hpp>

namespace Ui {
class SignUpDialog;
}

class SignUpDialog : public QDialog
{
    Q_OBJECT
    
public:
    explicit SignUpDialog(QWidget *parent = 0);
    ~SignUpDialog();
    
    void setImg(const cv::Mat &img);
    
private:
    Ui::SignUpDialog *ui;
    
signals:
    void sendData(std::string name);
private slots:
    void on_pushButton_OK_clicked();
    void on_pushButton_Cancel_clicked();
};

#endif // SIGNUPDIALOG_H
