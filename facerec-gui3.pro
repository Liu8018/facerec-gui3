#-------------------------------------------------
#
# Project created by QtCreator 2019-07-02T18:41:13
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = facerec-gui3
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += c++11

SOURCES += \
        FaceDetection/FaceDetection.cpp \
    FaceDetection/facedetectcnn-int8data.cpp \
    FaceDetection/facedetectcnn-model.cpp \
    FaceDetection/facedetectcnn.cpp \
        FaceRecognition/FaceRecognition.cpp \
        FeatExtraction/FeatExtraction.cpp \
        elm/ELM_functions.cpp \
        elm/elm_in_elm_model.cpp \
        elm/elm_model.cpp \
        functions.cpp \
        gui/mainwindow.cpp \
        main.cpp \
    gui/SignUpDialog.cpp \
    params.cpp

HEADERS += \
        FaceDetection/FaceDetection.h \
    FaceDetection/facedetectcnn.h \
        FaceRecognition/FaceRecognition.h \
        FeatExtraction/FeatExtraction.h \
        elm/ELM_functions.h \
        elm/elm_in_elm_model.h \
        elm/elm_model.h \
        functions.h \
        gui/mainwindow.h \
        params.h \
    gui/SignUpDialog.h

FORMS += \
        gui/mainwindow.ui \
    gui/SignUpDialog.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

INCLUDEPATH += $$PWD/../include
LIBS += -L$$PWD/../libs/ \
        -ldlib \
        -lpthread \
        -lopencv_core -lopencv_highgui -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs

RESOURCES += \
    gui/resources.qrc
