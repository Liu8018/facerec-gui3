#ifndef PARAMS_H
#define PARAMS_H

#include <opencv2/core.hpp>

//视频文件
extern std::string VIDEO_FILE;

//识别方法
extern std::string REC_METHOD;

//人脸数据库路径
const std::string FACEDB_PATH = "./data/face_database";

//人脸检测
const std::string FACEDT_MODEL_PATH = "./data/models/opencv_face_detector_uint8.pb";
const std::string FACEDT_MODELCONF_PATH = "./data/models/opencv_face_detector.pbtxt";

//特征提取
const std::string SHAPE_PREDICTOR_PATH = "./data/models/shape_predictor_68_face_landmarks.dat";
const std::string RESNET_MODEL_PATH = "./data/models/dlib_face_recognition_resnet_model_v1.dat";
const std::string RESNET_FEATS_PATH = "./data/face_database/resnetFeats.xml";
const std::string FEATS_PATH = "./data/face_database/feats.xml";
const std::string HASH_FILE_PATH = FACEDB_PATH+"/hashFile.dat";

//ELM模型
const std::string ELM_MODEL_PATH = "./data/models/ELM_Models";

#endif // PARAMS_H
