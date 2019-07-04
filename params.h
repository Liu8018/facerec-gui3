#ifndef PARAMS_H
#define PARAMS_H

#include <opencv2/core.hpp>

extern const std::string FACEDB_PATH;

extern const float FACEDT_CONF_TH;
extern const std::string FACEDT_MODEL_PATH;
extern const std::string FACEDT_MODELCONF_PATH;

extern const std::string FEATEX_METHOD;
extern const std::string SHAPE_PREDICTOR_PATH;
extern const std::string RESNET_MODEL_PATH;

extern const std::string ELM_MODEL_PATH;
extern const cv::Size FACE_IMGSIZE;
extern const int ELM_MODELS_COUNT;
extern const int ELM_NHIDDENNODES;

#endif // PARAMS_H
