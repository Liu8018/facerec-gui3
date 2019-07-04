#include "gui/mainwindow.h"
#include <QApplication>
#include "functions.h"
#include "params.h"
#include <iostream>

int main(int argc, char *argv[])
{
    std::string strArgv(argv[1]);
    std::cout<<"strArgv:"<<strArgv<<std::endl;
    if(strArgv.find("updatedb") == std::string::npos)
    {
        VIDEO_FILE = argv[2];
        std::cout<<"VIDEO_FILE:"<<VIDEO_FILE<<std::endl;
    }

    if(strArgv == "resnet" || strArgv == "elm")
    {
        REC_METHOD = strArgv;
        std::cout<<"REC_METHOD:"<<REC_METHOD<<std::endl;
    }
    else if(strArgv == "updatedb")
    {
        std::cout<<"mmp1"<<std::endl;
        handleFaceDb(1);
        handleFaceDb(2);
        return 0;
    }
    else if(strArgv == "updatedb-elm")
    {
        std::cout<<"mmp2"<<std::endl;
        handleFaceDb(1);
        return 0;
    }
    else if(strArgv == "updatedb-resnet")
    {
        std::cout<<"mmp3"<<std::endl;
        handleFaceDb(2);
        return 0;
    }
    else
    {
        std::cout<<"parameters error!"<<std::endl;
        return 0;
    }
    
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    
    return a.exec();
}
