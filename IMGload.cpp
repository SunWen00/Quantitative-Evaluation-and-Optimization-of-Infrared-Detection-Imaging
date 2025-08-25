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
    // 构建文件名
    std::string matname = mubiao + "_" + puduan + "_" + guangzhao + "_" +
        zhouye + "_rad_" + yun + ".mat";
    std::cout << matname << std::endl;

    // 打开.mat文件
    MATFile* pmat = matOpen(matname.c_str(), "r");
    if (pmat == nullptr) {
        std::cerr << "Error opening file: " << matname << std::endl;
        return cv::Mat();
    }

    // 读取total_radt变量
    mxArray* arr = matGetVariable(pmat, "total_radt");
    if (arr == nullptr) {
        std::cerr << "Error reading variable 'total_radt'" << std::endl;
        matClose(pmat);
        return cv::Mat();
    }

    // 获取数组维度
    mwSize ndims = mxGetNumberOfDimensions(arr);
    const mwSize* dims = mxGetDimensions(arr);

    // 检查帧数是否有效
    if (framenow < 1 || framenow > dims[2]) {
        std::cerr << "Invalid frame number: " << framenow << std::endl;
        mxDestroyArray(arr);
        matClose(pmat);
        return cv::Mat();
    }

    // 获取数据指针
    double* data = mxGetPr(arr);

    // 创建OpenCV矩阵（注意MATLAB是列优先，需要转置）
    cv::Mat total_radt(dims[0], dims[1], CV_64FC(dims[2]), data);
    cv::Mat frame;
    total_radt.slice(2, framenow - 1, 1).copyTo(frame);  // 提取指定帧
    frame = frame.t();  // 转置以匹配MATLAB的显示方向

    // 清理资源
    mxDestroyArray(arr);
    matClose(pmat);

    return frame;
}

// 示例用法
int main() {
    cv::Mat Jimgnow = ImgLoad("f22", "mwir", "nosun", "sea301", "cloud", 1);

    if (!Jimgnow.empty()) {
        // 归一化显示
        cv::Mat displayImg;
        cv::normalize(Jimgnow, displayImg, 0, 255, cv::NORM_MINMAX, CV_8U);

        cv::imshow("Image", displayImg);
        cv::waitKey(0);
    }

    return 0;
}