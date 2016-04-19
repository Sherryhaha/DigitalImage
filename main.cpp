#include <cv.h>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <math.h>
#include<vector>
#include<time.h>

using namespace std;
using namespace cv;

#define PI 3.1415926535
typedef uchar tc;
typedef double ty;

/**
 * 无偏处理
 */
void edifference(Mat &src) {

}

//计算图片的信息熵
double Entropy(Mat &A) {
    int En[256];
    double fpEn[256];
    double result, sum = 0, tmp;
    int X_image, Y_image;
    X_image = A.cols;
    Y_image = A.rows;
    int size = X_image * Y_image;
    int i, j;
    memset(&En, 0, sizeof(int) * 256);
    memset(&fpEn, 0, sizeof(double) * 256);
    for (i = 0; i < Y_image; i++) //计算差分矩阵直方图
    {
        for (j = 0; j < X_image; j++) {
            tc GrayIndex = A.at<tc>(i, j);
            En[GrayIndex]++;
        }
    }

    for (i = 0; i < 256; i++)   // 计算灰度分布密度
    {
        fpEn[i] = (double) En[i] / (double) size;
    }
    for (i = 0; i < 256; i++) {
        if (fpEn[i] > 0) {
            tmp = log(fpEn[i]) / log(2);
            sum = sum + fpEn[i] * tmp;
        }
    }
    sum = -sum;
    return sum;
}

//计算并输出图像的均值，标准差，信息熵
void MeanStdEntropy(Mat &A, string name) {
    //计算图像信息熵
    double entropy;
    entropy = Entropy(A);

    //计算图像方差与均值
    Scalar mean;
    Scalar stddev;

    meanStdDev(A, mean, stddev);
    ty mean_pxl = mean.val[0];
    ty stddev_pxl = stddev.val[0];
    cout << name << "均值：" << mean_pxl << " " + name << "的标准差：" << stddev_pxl << " " + name << "的信息熵：" << entropy <<
    endl;
    return;
}


/**
 * @function：makeGaussKern
 * @description:生成二维高斯模版
 * @input:标准差
 */
float *makeGaussKern(double sd) {
    double sum = 0, dvalue;
    double IDY, IDX;
    int tHeight = 1 + 2 * ceil(3 * sd);
    int tWidth = 1 + 2 * ceil(3 * sd);
    int tcenterY = (int) tHeight / 2;
    int tcenterX = (int) tWidth / 2;
    float *kernel;
    kernel = new float[tHeight * tWidth];
    double A = -1 / (2 * sd * sd);
    double B = -A / PI;
    for (int i = 0; i < tHeight; i++) {
        for (int j = 0; j < tWidth; j++) {
            IDY = abs(i - tcenterY);
            IDX = abs(j - tcenterX);
            dvalue = B * exp(A * (IDX * IDX + IDY * IDY));
            kernel[i * tWidth + j] = dvalue;
            sum += dvalue;
        }
    }

//    for(int i = 0;i<tHeight;i++){
//        for(int j = 0;j<tWidth;j++){
//            kernel[i*tWidth + j] /=sum;
//            cout<<kernel[i*tWidth+j]<<" ";
//        }
//        cout<<endl;
//    }
    return kernel;
}

/**
 * @function:Gauss2D
 * @description:灰度图二维高斯模糊
 * @input:原图像、目标图像、二维高斯模板、标准差大小
 */
void Gauss2D(Mat &src, Mat &src1, float *kernel, double segma) {
    int pHeight = src.rows;
    int pWidth = src.cols;
    int tHeight = 1 + 2 * ceil(3 * segma);
    int tWidth = 1 + 2 * ceil(3 * segma);
    int tcenterY = (int) tHeight / 2;
    int tcenterX = (int) tWidth / 2;
    double pResult;
    src1.create(src.size(), src.type());
    CV_Assert(src.depth() != sizeof(uchar));
    for (int i = 0; i < pHeight - tHeight + 1; i++) {
        for (int j = 0; j < pWidth - tWidth + 1; j++) {
            pResult = 0;
            for (int k = 0; k < tHeight; k++) {
                for (int l = 0; l < tWidth; l++) {
                    pResult += src.at<float>(i + k, j + l) * kernel[k * tHeight + l];
                }
            }
            if (pResult > 255) {
                src1.at<float>(i + tHeight / 2, j + tHeight / 2) = 255;
            }
            else if (pResult < 0) {
                src1.at<float>(i + tHeight / 2, j + tHeight / 2) = 0;
            }
            else {
                src1.at<float>(i + tHeight / 2, j + tHeight / 2) = pResult;
            }
        }
    }

}

/**
 * @function:Gauss
 * @description:实现灰度图的一维高斯模糊
 * @input:原图像、目标图像、方差
 */
void Gauss(Mat &src, Mat &dst, double segma) {
    int ksize;
    ksize = 1 + 2 * ceil(3 * segma);
    int kcenter = ksize / 2;

    //  double *skernel;
    //  skernel= new double(ksize);
    //  double skernel[100000];
    vector<double> skernel;
    double sum = 0, psum = 0;
    double dvalue = 0;
    double aa = -1 / (2 * segma * segma);
    double pi2 = 2 * PI;
    double bb = sqrt(pi2) * segma;
    double g, d, distance;
    int i = 0, j = 0;
    Mat tmp;
    tmp.create(src.size(), src.type());
    dst.create(src.size(), src.type());

    if (src.channels() != 1) {
        cout << "hhh" << endl;
        return;
    }
    for (i = 0; i < ksize; i++) {
        d = i - kcenter;
        distance = d * d;
//        *(skernel+i) = exp(aa*distance)/bb;
        skernel.push_back(exp(aa * distance) / bb);
        sum += skernel[i];
    }

    for (i = 0; i < ksize; i++) {
        skernel[i] = skernel[i] / sum;
    }

    //x方向的高斯模糊
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            psum = 0;
            dvalue = 0;
            for (i = -kcenter; i <= kcenter; i++) {
                if ((x + i) >= 0 && (x + i) < src.cols) {
                    dvalue += src.at<float>(y, x + i) * (skernel[kcenter + i]);
                    psum += skernel[kcenter + i];
                }

            }

            tmp.at<float>(y, x) = dvalue / psum;
        }
    }

    //y方向的高斯模糊
    for (int x = 0; x < src.cols; x++) {
        for (int y = 0; y < src.rows; y++) {
            psum = 0;
            dvalue = 0;
            for (i = -kcenter; i <= kcenter; i++) {
                if ((y + i) >= 0 && (y + i) < src.rows) {
                    dvalue += tmp.at<float>(y + i, x) * (skernel[kcenter + i]);
                    psum += skernel[kcenter + i];
                }
            }

            dst.at<float>(y, x) = dvalue / psum;
        }
    }
    // delete[] skernel;
    // skernel = NULL;
}

/**
 * function:CTlog
 * description:convert image to log
 */
void CTlog(Mat &src, Mat &dst) {
    int Height = src.cols;
    int Width = src.rows;
    int x, y;
    double dvalue;
    for (x = 0; x < Height; x++) {
        for (y = 0; y < Width; y++) {
            dvalue = log(src.at<float>(x, y));
            dst.at<float>(x, y) = dvalue;
        }

    }
}

/**
 * function:subImg
 * description:将两个同样大小的图片相减
 */
void subImg(Mat &src1, Mat &src2, Mat &dst) {
    int Height = src1.cols;
    int Width = src1.rows;
    int x, y;
    int dvalue;
    for (x = 0; x < Height; x++) {
        for (y = 0; y < Width; y++) {
            dst.at<float>(x, y) = src1.at<float>(x, y) - src2.at<float>(x, y);
        }
    }
}

/**
 * function:stretch
 * descrition:将图像从对数域转换到空间域
 */
void stretch(Mat &src) {
    int Height = src.cols;
    int Width = src.rows;
    cout << Height << "--" << Width << endl;
    int x = 0, y = 0;
    double lmax = 0, lmin = 255;
    double min, max, maxmin, dvalue, avg = 0, fc = 0;
    double dli;
    int scale = 2;
    for (x = 0; x < Height; x++) {
        for (y = 0; y < Width; y++) {
            dli = src.at<float>(y, x);
            if (dli > lmax) {
                lmax = dli;
            }
            if (dli < lmin) {
                lmin = dli;
            }
            avg += dli;
        }
    }
    avg = avg / (Height * Width);
    dli = 0;
    for (x = 0; x < Height; x++) {
        for (y = 0; y < Width; y++) {
            dli = src.at<float>(y, x);
            fc = fc + (dli - avg) * (dli - avg);
        }
    }
    fc = fc / (Height * Width);

    Scalar me, de;
    meanStdDev(src, me, de);
    min = me.val[0] - scale * de.val[0];
    max = me.val[0] + scale * de.val[0];
    maxmin = max - min;
    for (x = 0; x < Height; x++) {
        for (y = 0; y < Width; y++) {
            dvalue = ((src.at<float>(y, x) - min) / maxmin);
            src.at<float>(y, x) = dvalue;
        }
    }
}

/**
 * function:pImg
 * description:打印出图像的各个像素值
 */
void pImg(Mat &src) {
    int Height = src.cols;
    int Width = src.rows;
    int x = 0, y = 0;
    for (x = 0; x < Height; x++) {
        for (y = 0; y < Width; y++) {
            cout << src.at<float>(x, y) << " ";
        }
        cout << endl;
    }
}

/**
 * function:addforlog
 * description:将图像的每个像素值加f
 */
void addforlog(Mat &src, Mat &dst, float f) {
    int Height = src.cols;
    int Width = src.rows;
    int x = 0, y = 0;
    for (x = 0; x < Height; x++) {
        for (y = 0; y < Width; y++) {
            dst.at<float>(y, x) = src.at<float>(y, x) + f;
        }
    }
}

/**
 * function:sunSSR
 * description:将图像进行SS（单尺度Retinex增强）
 */
void sunSSR(Mat &src, Mat &dst, double segma) {
    Mat src_s, src_r;
    src_s.create(src.size(), CV_32F);
    dst.create(src.size(), CV_32F);
//  将原图转换到对数域，将每个像素都要加一，保证对数运算正确

    addforlog(src, dst, 1.0);
//    pImg(dst);
    log(dst, src_s);
//  二维高斯模糊
//  int l = 1+2*ceil(3*segma);
//  float *Gkernel = new float(l*l);
//  Gkernel = makeGaussKern(segma);
//  Gauss2D(src,src_r,Gkernel,segma);
//  一维高斯模糊
    Gauss(src, src_r, segma);
//  CTlog(src_r,src_r);
    addforlog(src_r, src_r, 1.0);
    log(src_r, src_r);
//  subImg(src_s,src_r,dst);
//  logL = logR-logS
    subtract(src_s, src_r, dst);
    stretch(dst);
    return;
}


int main() {
//  执行retinex时用30比较好，实验时先写5
    double segma = 200;
    int l = 1 + 2 * ceil(3 * segma);
    string filename = "/Users/sunguoyan/Downloads/picture/lenamo.jpg";

    Mat src, src1, dst, dst1;
    src = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    src.convertTo(src1, CV_32F, 1 / 255.0);
    dst1.create(src.size(), src.type());

    clock_t start = clock();
    double totaltime;

//  二维高斯模糊
//  float *Gkernel = new float(l*l);
//  Gkernel = makeGaussKern(segma);
//  Gauss2D(src1,dst,Gkernel,segma);
//
//    dst.create(src.size(), CV_32F);
//    Gauss(src1,dst,segma);
//  log(src1,dst);
    sunSSR(src1, dst, segma);
    clock_t end = clock();
    totaltime = (double) (end - start) / CLOCKS_PER_SEC;
    cout << "运行时间: " << totaltime << endl;
//    FileStorage fs("/Users/sunguoyan/Downloads/picture/lenamo.bmp", FileStorage::WRITE);
//    fs<<"lenamo"<<dst;


//    imwrite("/Users/sunguoyan/Downloads/picture/lenamo.bmp",dst1);

    dst.convertTo(dst1, CV_8U, 255, 0);

    MeanStdEntropy(src, "原图像");
    MeanStdEntropy(dst1, "Retinex增强后图像");

    namedWindow("test");
    namedWindow("test1");
    imshow("test", src1);
    imshow("test1", dst1);
    waitKey(0);
//    fs.release();
    return 0;
}