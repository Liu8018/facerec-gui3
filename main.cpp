#include "gui/mainwindow.h"
#include <QApplication>
#include "functions.h"
#include "params.h"
#include <iostream>

int main(int argc, char *argv[])
{
    std::string strArgv(argv[1]);
    if(strArgv.find("updatedb") == std::string::npos)
    {
        if(argc < 3)
        {
            std::cout<<"parameters error!"<<std::endl;
            return 0;
        }
        
        VIDEO_FILE = argv[2];
    }

    if(strArgv == "resnet" || strArgv == "elm")
    {
        REC_METHOD = strArgv;
    }
    else if(strArgv == "updatedb")
    {
        handleFaceDb(1);
        handleFaceDb(2);
        return 0;
    }
    else if(strArgv == "updatedb-elm")
    {
        handleFaceDb(1);
        return 0;
    }
    else if(strArgv == "updatedb-resnet")
    {
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
