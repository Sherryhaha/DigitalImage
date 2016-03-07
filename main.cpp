
#include <cv.h>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <math.h>
using namespace std;
using namespace cv;

#define PI 3.1415926535

/********************************************************************************
                                单尺度Retinex图像增强程序
                                src为待处理图像
                                sd为高斯模糊标准差
                                scale为对比度系数
*********************************************************************************/

//生成二维高斯模板
float *makeGaussKern(double sd){
    double sum = 0,dvalue;
    double IDY,IDX;
    int tHeight = 1+2*ceil(3*sd);
    int tWidth = 1+2*ceil(3*sd);
    int tcenterY = (int)tHeight/2;
    int tcenterX = (int)tWidth/2;
    float *kernel;
    kernel = new float[tHeight*tWidth];
    double A = -1/(2*sd*sd);
    double B = -A/PI;
    for(int i = 0;i < tHeight; i++){
        for(int j = 0;j < tWidth; j++){
            IDY = abs(i - tcenterY);
            IDX = abs(j - tcenterX);
            dvalue = B*exp(A*(IDX*IDX+IDY*IDY));
            kernel[i * tWidth + j] = dvalue;
            sum+=dvalue;
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
//灰度图二维高斯模糊
void Gauss2D(Mat &src,Mat &src1,float *kernel,double segma){
    int pHeight = src.rows;
    int pWidth = src.cols;
    int tHeight = 1+2*ceil(3*segma);
    int tWidth = 1+2*ceil(3*segma);
    int tcenterY = (int)tHeight/2;
    int tcenterX = (int)tWidth/2;
    double pResult;
    uchar* p,*p1;
    src1.create(src.size(),src.type());
    //src.data代表src的数据部分的指针
    uchar* srcData = src.data;
    uchar* dstData = src1.data;
    //src1.create(pHeight,pWidth,CV_8UC1);
  //  cout<<src.step<<"hehe"<<endl;
   // cout<<src.cols<<"hehe"<<endl;
    CV_Assert(src.depth() != sizeof(uchar));
    for(int i = 0;i < pHeight - tHeight+1 ;i++){
        p = src.ptr<uchar>(i);
        p1 = src1.ptr<uchar>(i);
        for(int j = 0;j<pWidth - tWidth+1 ;j++){
            pResult = 0;
           // double acc = 0;
            for(int k = 0;k < tHeight;k++){

                for(int l =0;l<tWidth;l++){
                   // pResult += p[j]*kernel[k*tHeight+l];
                    pResult+=src.at<uchar>(i+k,j+l)*kernel[k*tHeight+l];

                   // acc += *(srcData + src.step * (i+l) + (j+k)) * kernel[k*tHeight+l];
                }
            }
            if(pResult > 255){
               // *(dstData + src1.step * (i+tHeight/2)  + (j+tWidth/2) ) = 255;
                src1.at<uchar>(i+tHeight/2,j+tHeight/2) = 255;
            }
            else if(pResult < 0){
                //*(dstData + src1.step * (i+tHeight/2)  + (j+tWidth/2) ) = 0;
                src1.at<uchar>(i+tHeight/2,j+tHeight/2) = 0;
            }
            else{
               // *(dstData + src1.step * (i+tHeight/2)  + (j+tWidth/2) ) = (int)acc;
                src1.at<uchar>(i+tHeight/2,j+tHeight/2) = (int)pResult;
            }
        }
    }

}

// 灰度图一维高斯模板
void Gauss(Mat &src,Mat &dst,double segma){
    int ksize;
    ksize = 1+2*ceil(3*segma);
    int kcenter = ksize/2;
    double *kernel = new double(ksize);
    int sum = 0;
    double dvalue = 0;
    double aa = -1/(2*segma*segma);
    double bb = 1/(sqrt(2*PI*segma*segma));
    double g,d;
    Mat tmp;
    tmp.create(src.size(),src.type());
    if(src.channels()!=1){
        return;
    }
    for(int i = 0;i < ksize;i++){
        d = i - kcenter;
        g = bb*exp(aa*d*d);
        kernel[i] = g;
        sum+=kernel[i];
    }

    for(int i = 0;i<ksize;i++){
        kernel[i] = kernel[i]/sum;
    }

    dst.create(src.size(),src.type());
    //x方向的高斯模糊
    for(int y = 0;y < src.rows;y++){
        for(int x = 0;x < src.cols;x++){
            for(int i = -kcenter;i<kcenter;i++){
                if(x+i>0&&x+i<src.cols){
                    dvalue += src.at<uchar>(y,x+i)*kernel[kcenter+i];
                    sum += kernel[kcenter+i];
                }

            }

            tmp.at<uchar>(x,y) = dvalue/sum;
        }
    }

    //y方向的高斯模糊
    for(int x = 0;x < src.cols;x++){
        for(int y = 0;y < src.rows;y++){
            for(int i = -kcenter;i<kcenter;i++){
                if(y+i>0&&y+i<src.rows){
                    dvalue += tmp.at<uchar>(y+i,x)*kernel[kcenter+i];
                    sum += kernel[kcenter+i];
                }

            }

            dst.at<uchar>(x,y) = dvalue/sum;
        }
    }

}

void SSR(IplImage* src,int sigma,int scale)
{
    IplImage* src_fl  = cvCreateImage(cvGetSize(src),IPL_DEPTH_32F,src->nChannels);
    IplImage* src_fl1 = cvCreateImage(cvGetSize(src),IPL_DEPTH_32F,src->nChannels);
    IplImage* src_fl2 = cvCreateImage(cvGetSize(src),IPL_DEPTH_32F,src->nChannels);
    float a=0.0,b=0.0,c=0.0;
    cvConvertScale(src,src_fl,1.0,1.0);//转换范围，所有图像元素增加1.0保证cvlog正常
    cvLog(src_fl,src_fl1);

    cvSmooth(src_fl,src_fl2,CV_GAUSSIAN,0,0,sigma);        //SSR算法的核心之一，高斯模糊

    cvLog(src_fl2,src_fl2);
    cvSub(src_fl1,src_fl2,src_fl);//Retinex公式，Log(R(x,y))=Log(I(x,y))-Log(Gauss(I(x,y)))

    //计算图像的均值、方差，SSR算法的核心之二
    //使用GIMP中转换方法：使用图像的均值方差等信息进行变换
    //没有添加溢出判断
    CvScalar mean;
    CvScalar dev;
    cvAvgSdv(src_fl,&mean,&dev,NULL);//计算图像的均值和标准差
    double min[3];
    double max[3];
    double maxmin[3];
    for (int i=0;i<3;i++)
    {
        min[i]=mean.val[i]-scale*dev.val[i];
        max[i]=mean.val[i]+scale*dev.val[i];
        maxmin[i]=max[i]-min[i];
    }
    float* data2=(float*)src_fl->imageData;
    for (int i=0;i<src_fl2->width;i++)
    {
        for(int j=0;j<src_fl2->height;j++)
        {
            data2[j*src_fl->widthStep/4+3*i+0]=255*(data2[j*src_fl->widthStep/4+3*i+0]-min[0])/maxmin[0];
            data2[j*src_fl->widthStep/4+3*i+1]=255*(data2[j*src_fl->widthStep/4+3*i+1]-min[1])/maxmin[1];
            data2[j*src_fl->widthStep/4+3*i+2]=255*(data2[j*src_fl->widthStep/4+3*i+2]-min[2])/maxmin[2];
        }
    }


    cvConvertScale(src_fl,src,1,0);
    cvReleaseImage(&src_fl);
    cvReleaseImage(&src_fl1);
    cvReleaseImage(&src_fl2);
}

int main()
{

    double segma = 5;
    int l = 1+2*ceil(3*segma);
    string filename="/Users/sunguoyan/Downloads/picture/lena.bmp";
  //  float *Gkernel = new float(l*l);
    Mat src,src1;
    src = imread(filename,0);


   // Gkernel = makeGaussKern(segma);



   // Gauss2D(src,src1,Gkernel,segma);

    Gauss(src,src1,segma);
    namedWindow("test");
    namedWindow("test1");
    imshow("test",src);

    imshow("test1",src1);
    waitKey(0);

//    IplImage* frog=cvLoadImage(filename,1);
//    IplImage* frog1=cvCreateImage(cvGetSize(frog),IPL_DEPTH_32F,frog->nChannels);
//    cvConvertScale(frog,frog1,1.0/255,0);
//    SSR(frog,30,2);
//    cvNamedWindow("hi");
//    cvNamedWindow("hi1");
//    cvShowImage("hi1",frog1);
//    cvShowImage("hi",frog);
//    cvWaitKey(0);
//    cvDestroyAllWindows();
//    cvReleaseImage(&frog);
//    cvReleaseImage(&frog1);


    return 0;
}