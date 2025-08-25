#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace chrono;

// 图像加载函数
Mat imgLoad(const string& object, const string& band, const string& ze,
    const string& sea, const string& cloud, int param) {
  
    Mat img(256, 256, CV_64F);
    randn(img, 0, 0.1); // 生成随机测试数据
    return img;
}

// 计算信噪比SNR
double calculateSNR(const Mat& img) {
    Scalar mean, stddev;
    meanStdDev(img, mean, stddev);
    return mean[0] / stddev[0];
}

// 成像处理函数
pair<VectorXd, Mat> chengxiangTarget(double koujing, double jiaoju, double tancejuli,
    double guangxuexiaolv, double boduan, double nengliang,
    double yiqibeijing, double litijiao, double xiangyuan,
    double andianliu, double jifenshijian, double duchuzaosheng,
    double manjingdianzi, double lianghua, double liangzixiaolv) {

    Mat img(256, 256, CV_64F);
    randn(img, 0.5, 0.1); // 生成模拟图像

    // 生成评估参数向量
    VectorXd params(12);
    params << koujing, jiaoju, guangxuexiaolv, nengliang, yiqibeijing,
        xiangyuan, andianliu, jifenshijian, manjingdianzi,
        liangzixiaolv, 0.0, 0.0; 

    return { params, img };
}

// 数据正向化处理
MatrixXd positivization(const MatrixXd& X, const vector<int>& position, const vector<int>& type) {
    MatrixXd result = X;
    int n = X.rows();

    for (int i = 0; i < position.size(); ++i) {
        int col = position[i] - 1; // 转换为0基索引
        int t = type[i];
        VectorXd colData = X.col(col);

        if (t == 1) { // 极小型
            double maxVal = colData.maxCoeff();
            result.col(col) = maxVal - colData;
        }
    }

    return result;
}

// 熵权法计算权重
VectorXd entropyMethod(const MatrixXd& Z) {
    int n = Z.rows();
    int m = Z.cols();

    MatrixXd P(n, m);
    for (int j = 0; j < m; ++j) {
        double sum = Z.col(j).sum();
        P.col(j) = Z.col(j) / sum;
    }

    VectorXd e(m);
    for (int j = 0; j < m; ++j) {
        double sum = 0;
        for (int i = 0; i < n; ++i) {
            if (P(i, j) > 0) {
                sum += P(i, j) * log(P(i, j));
            }
        }
        e(j) = -sum / log(n);
    }

    VectorXd w = (VectorXd::Ones(m) - e) / (m - e.sum());
    return w;
}

int main() {
    // 记录开始时间
    auto start = system_clock::now();
    time_t startTime = system_clock::to_time_t(start);
    cout << "开始时间: " << ctime(&startTime) << endl;

    // 读取图像
    Mat IRrad = imgLoad("f22", "mwir", "ze20", "sea301", "cloud", 1);
    imshow("Original Image", IRrad / IRrad.maxVal());
    waitKey(100);

    // 设置目标参数
    int target = 1;
    vector<double> AndianliuInput = { 6e-7, 4e-7, 4e-7, 3e-7, 6e-7, 7e-7 };
    vector<double> YiqibeijingInput = { 70e-3, 55e-3, 50e-3, 45e-3, 55e-3, 65e-3 };
    vector<double> KoujingInput = { 0.5, 0.55, 0.4, 0.5, 0.4, 0.6 };
    vector<double> JiaojuInput = { 1.2, 1.12, 1.2, 1.16, 1.18, 1.3 };
    vector<double> LiangzixiaolvInput = { 0.35, 0.36, 0.37, 0.35, 0.36, 0.35 };
    vector<double> GuangxuexiaolvInput = { 0.5, 0.59, 0.57, 0.45, 0.55, 0.54 };
    vector<double> JifenshijianInput = { 1, 1, 1, 1, 1, 1 };
    vector<double> NengliangjizhongduInput = { 0.5, 0.45, 0.52, 0.55, 0.6, 0.49 };
    vector<double> XiangyuanchicunInput = { 15, 16, 18, 16, 17, 16 };
    vector<double> ManjingdianzishuInput = { 6, 6, 6, 6, 6, 6 };

    // 其他固定参数（根据实际情况设置）
    double TancejuliInput = 1000;
    double BoduanInput = 5.0;
    double LitijiaoInput = 0.5;
    double DuchuzaoshengInput = 1e-8;
    double LianghuaweishuInput = 0.8;

    int n = AndianliuInput.size();
    MatrixXd X(n, 12);
    vector<double> snr(n);
    vector<Mat> IRIMG;

    // 循环处理每个参数组合
    for (int i = 0; i < n; ++i) {
        auto [params, img] = chengxiangTarget(
            KoujingInput[i], JiaojuInput[i], TancejuliInput,
            GuangxuexiaolvInput[i], BoduanInput, NengliangjizhongduInput[i],
            YiqibeijingInput[i], LitijiaoInput, XiangyuanchicunInput[i],
            AndianliuInput[i], JifenshijianInput[i], DuchuzaoshengInput,
            ManjingdianzishuInput[i], LianghuaweishuInput, LiangzixiaolvInput[i]
        );

        X.row(i) = params;
        snr[i] = calculateSNR(img);
        IRIMG.push_back(img);

        // 保存图像
        Mat normalized;
        normalize(img, normalized, 0, 255, NORM_MINMAX, CV_8U);
        imwrite("result_" + to_string(i) + ".png", normalized);
    }

    // 数据正向化处理
    vector<int> Position = { 1, 11 }; // 需要处理的列（1基）
    vector<int> Type = { 1, 1 };      // 极小型
    MatrixXd X_pos = positivization(X, Position, Type);

    // 标准化处理
    int m = X_pos.cols();
    MatrixXd Z(n, m);
    for (int j = 0; j < m; ++j) {
        double norm = X_pos.col(j).norm();
        Z.col(j) = X_pos.col(j) / norm;
    }

    // 计算权重
    bool Judge = true;
    VectorXd W;
    if (Judge) {
        W = entropyMethod(Z);
    }
    else {
        W = VectorXd::Ones(m) / m;
    }

    // 计算TOPSIS得分
    VectorXd maxZ = Z.colwise().maxCoeff();
    VectorXd minZ = Z.colwise().minCoeff();

    VectorXd D_P(n), D_N(n);
    for (int i = 0; i < n; ++i) {
        D_P(i) = sqrt(((Z.row(i) - maxZ).array().square() * W.array()).sum());
        D_N(i) = sqrt(((Z.row(i) - minZ).array().square() * W.array()).sum());
    }

    VectorXd S = D_N.array() / (D_P.array() + D_N.array());
    double sumS = S.sum();
    VectorXd stand_S = S / sumS;

    // 排序结果
    vector<int> index(n);
    iota(index.begin(), index.end(), 0);
    sort(index.begin(), index.end(), [&](int a, int b) {
        return stand_S(a) > stand_S(b);
        });

    // 输出结果
    cout << "最终得分: " << endl;
    for (int i = 0; i < n; ++i) {
        cout << "方案 " << index[i] + 1 << ": " << stand_S(index[i]) << endl;
    }

    // 按SNR排序
    vector<int> index1(n);
    iota(index1.begin(), index1.end(), 0);
    sort(index1.begin(), index1.end(), [&](int a, int b) {
        return snr[a] > snr[b];
        });

    cout << "\nSNR排序: " << endl;
    for (int i = 0; i < n; ++i) {
        cout << "方案 " << index1[i] + 1 << ": " << snr[index1[i]] << endl;
    }

    // 记录结束时间
    auto end = system_clock::now();
    duration<double> elapsed = end - start;
    cout << "\n程序运行时间: " << elapsed.count() << " 秒" << endl;

    return 0;
}
