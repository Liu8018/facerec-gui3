#include "params.h"

//人脸数据库路径
const std::string FACEDB_PATH = "./data/face_database";

//人脸检测
const float FACEDT_CONF_TH = 0.7;
const std::string FACEDT_MODEL_PATH = "./data/models/opencv_face_detector_uint8.pb";
const std::string FACEDT_MODELCONF_PATH = "./data/models/opencv_face_detector.pbtxt";

//特征提取
const std::string FEATEX_METHOD = "resnet";
const std::string SHAPE_PREDICTOR_PATH = "./data/models/shape_predictor_68_face_landmarks.dat";
const std::string RESNET_MODEL_PATH = "./data/models/dlib_face_recognition_resnet_model_v1.dat";

//ELM模型
const std::string ELM_MODEL_PATH = "./data/models/ELM_Models";
const cv::Size FACE_IMGSIZE = cv::Size(50,50);//cv::Size(w,h)
const int ELM_MODELS_COUNT = 10;
const int ELM_NHIDDENNODES = 100;
