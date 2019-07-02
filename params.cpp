#include "params.h"

const char* FACEDB_PATH = "./data/face_database";

const float FACEDT_CONF_TH = 0.7;
const char* FACEDT_MODEL_PATH = "./data/models/opencv_face_detector_uint8.pb";
const char* FACEDT_MODELCONF_PATH = "./data/models/opencv_face_detector.pbtxt";

const char* SHAPE_PREDICTOR_PATH = "./data/models/shape_predictor_68_face_landmarks.dat";
const char* RESNET_MODEL_PATH = "./data/models/dlib_face_recognition_resnet_model_v1.dat";

const char* ELM_MODEL_PATH = "./data/models/ELM_Models";
const cv::Size FACE_IMGSIZE = cv::Size(50,50);//cv::Size(w,h)
