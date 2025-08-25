#include <iostream>
#include <string>
#include <vector>
#include "mat.h"
#include "opencv2/opencv.hpp"

cv::Mat ImgLoad(const std::string& mubiao,
    const std::string& puduan,
    const std::string& guangzhao,
    const std::string& zhouye,
    const std::string& yun,
    int framenow) {
    // �����ļ���
    std::string matname = mubiao + "_" + puduan + "_" + guangzhao + "_" +
        zhouye + "_rad_" + yun + ".mat";
    std::cout << matname << std::endl;

    // ��.mat�ļ�
    MATFile* pmat = matOpen(matname.c_str(), "r");
    if (pmat == nullptr) {
        std::cerr << "Error opening file: " << matname << std::endl;
        return cv::Mat();
    }

    // ��ȡtotal_radt����
    mxArray* arr = matGetVariable(pmat, "total_radt");
    if (arr == nullptr) {
        std::cerr << "Error reading variable 'total_radt'" << std::endl;
        matClose(pmat);
        return cv::Mat();
    }

    // ��ȡ����ά��
    mwSize ndims = mxGetNumberOfDimensions(arr);
    const mwSize* dims = mxGetDimensions(arr);

    // ���֡���Ƿ���Ч
    if (framenow < 1 || framenow > dims[2]) {
        std::cerr << "Invalid frame number: " << framenow << std::endl;
        mxDestroyArray(arr);
        matClose(pmat);
        return cv::Mat();
    }

    // ��ȡ����ָ��
    double* data = mxGetPr(arr);

    // ����OpenCV����ע��MATLAB�������ȣ���Ҫת�ã�
    cv::Mat total_radt(dims[0], dims[1], CV_64FC(dims[2]), data);
    cv::Mat frame;
    total_radt.slice(2, framenow - 1, 1).copyTo(frame);  // ��ȡָ��֡
    frame = frame.t();  // ת����ƥ��MATLAB����ʾ����

    // ������Դ
    mxDestroyArray(arr);
    matClose(pmat);

    return frame;
}

// ʾ���÷�
int main() {
    cv::Mat Jimgnow = ImgLoad("f22", "mwir", "nosun", "sea301", "cloud", 1);

    if (!Jimgnow.empty()) {
        // ��һ����ʾ
        cv::Mat displayImg;
        cv::normalize(Jimgnow, displayImg, 0, 255, cv::NORM_MINMAX, CV_8U);

        cv::imshow("Image", displayImg);
        cv::waitKey(0);
    }

    return 0;
}