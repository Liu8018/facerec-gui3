#include "SignUpDialog.h"
#include "ui_SignUpDialog.h"

SignUpDialog::SignUpDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::SignUpDialog)
{
    ui->setupUi(this);
}

SignUpDialog::~SignUpDialog()
{
    delete ui;
}

void SignUpDialog::on_pushButton_OK_clicked()
{
    emit sendData(ui->lineEdit->text().toStdString());
    this->close();
}

void SignUpDialog::on_pushButton_Cancel_clicked()
{
    this->close();
}

void SignUpDialog::setImg(const cv::Mat &img)
{
    cv::Mat imgToShow;
    cv::resize(img,imgToShow,cv::Size(ui->label->width()-2,ui->label->height()-2));

    cv::cvtColor(imgToShow,imgToShow,cv::COLOR_BGR2RGB);
    QImage qimg = QImage((const uchar*)imgToShow.data,
                  imgToShow.cols,imgToShow.rows,
                  imgToShow.cols*imgToShow.channels(),
                  QImage::Format_RGB888);

    ui->label->setPixmap(QPixmap::fromImage(qimg));
}
