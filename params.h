#ifndef PARAMS_H
#define PARAMS_H

#include <opencv2/core.hpp>

//路径
extern const char* FACEDB_PATH;

//人脸检测
extern const float FACEDT_CONF_TH;//置信度
extern const char* FACEDT_MODEL_PATH;//模型路径
extern const char* FACEDT_MODELCONF_PATH;//模型config路径

//特征提取
extern const char* SHAPE_PREDICTOR_PATH;
extern const char* RESNET_MODEL_PATH;

//ELM模型
extern const char* ELM_MODEL_PATH;
extern const cv::Size FACE_IMGSIZE;

#endif // PARAMS_H
