#include "gui/mainwindow.h"
#include <QApplication>
#include "functions.h"
#include "params.h"
#include <iostream>

int main(int argc, char *argv[])
{
    std::string strArgv(argv[1]);
    
    if(strArgv == "clean")
    {
        std::string cmd_rm_elmModels = "rm " + ELM_MODEL_PATH + "/*";
        std::cout<<cmd_rm_elmModels<<std::endl;
        system(cmd_rm_elmModels.data());
        
        std::string cmd_rm_featsFiles = "rm " + FEATS_PATH + " " + HASH_FILE_PATH;
        std::cout<<cmd_rm_featsFiles<<std::endl;
        system(cmd_rm_featsFiles.data());
        
        std::string cmd_rm_resnetFeats = "rm " + RESNET_FEATS_PATH;
        std::cout<<cmd_rm_resnetFeats<<std::endl;
        system(cmd_rm_resnetFeats.data());
        
        return 0;
    }
    if(strArgv == "clean-elm")
    {
        std::string cmd_rm_elmModels = "rm " + ELM_MODEL_PATH + "/*";
        std::cout<<cmd_rm_elmModels<<std::endl;
        system(cmd_rm_elmModels.data());
        
        std::string cmd_rm_featsFile = "rm " + FEATS_PATH;
        std::cout<<cmd_rm_featsFile<<std::endl;
        system(cmd_rm_featsFile.data());
        
        std::string cmd_rm_pcaFile = "rm ./data/face_database/pca.xml";
        std::cout<<cmd_rm_pcaFile<<std::endl;
        system(cmd_rm_pcaFile.data());
        
        return 0;
    }
    if(strArgv == "clean-resnet")
    {
        std::string cmd_rm_resnetFeats = "rm " + RESNET_FEATS_PATH;
        std::cout<<cmd_rm_resnetFeats<<std::endl;
        system(cmd_rm_resnetFeats.data());
        
        return 0;
    }
    
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
