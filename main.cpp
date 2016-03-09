
#include <cv.h>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <math.h>
#include<vector>
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
   // uchar* p,*p1;
    src1.create(src.size(),CV_32F);
    //src.data代表src的数据部分的指针
   // uchar* srcData = src.data;
   // uchar* dstData = src1.data;
    //src1.create(pHeight,pWidth,CV_8UC1);
  //  cout<<src.step<<"hehe"<<endl;
   // cout<<src.cols<<"hehe"<<endl;
    CV_Assert(src.depth() != sizeof(uchar));
    for(int i = 0;i < pHeight - tHeight+1 ;i++){
     //   p = src.ptr<uchar>(i);
     //   p1 = src1.ptr<uchar>(i);
        for(int j = 0;j<pWidth - tWidth+1 ;j++){
            pResult = 0;
            for(int k = 0;k < tHeight;k++){
                for(int l =0;l<tWidth;l++){
                   // pResult += p[j]*kernel[k*tHeight+l];
                    pResult+=src.at<float>(i+k,j+l)*kernel[k*tHeight+l];
                   // acc += *(srcData + src.step * (i+l) + (j+k)) * kernel[k*tHeight+l];
                }
            }
            if(pResult > 255){
               // *(dstData + src1.step * (i+tHeight/2)  + (j+tWidth/2) ) = 255;
                src1.at<float>(i+tHeight/2,j+tHeight/2) = 255;
            }
            else if(pResult < 0){
                //*(dstData + src1.step * (i+tHeight/2)  + (j+tWidth/2) ) = 0;
                src1.at<float>(i+tHeight/2,j+tHeight/2) = 0;
            }
            else{
               // *(dstData + src1.step * (i+tHeight/2)  + (j+tWidth/2) ) = (int)acc;
                src1.at<float>(i+tHeight/2,j+tHeight/2) = pResult;
            }
        }
    }

}

// 灰度图一维高斯模糊
void Gauss(Mat &src,Mat &dst,double segma){
    int ksize;
    ksize = 1+2*ceil(3*segma);
    int kcenter = ksize/2;
   // double *skernel;
  //  skernel= new double(ksize);
  //  double skernel[100000];
    vector<double> skernel;
    double sum = 0,psum = 0;
    double dvalue = 0;
    double aa = -1/(2*segma*segma);
    double pi2 = 2*PI;
    double bb =  sqrt(pi2)*segma;
    double g,d,distance;
    int i = 0,j = 0;
    Mat tmp;
    tmp.create(src.size(),src.type());
    dst.create(src.size(),src.type());

    if(src.channels()!=1){
        cout<<"hhh"<<endl;
        return;
    }
    for( i = 0;i < ksize;i++){
        d = i - kcenter;
        distance = d*d;
//        cout<<"距离"<<distance<<" ";
//        *(skernel+i) = exp(aa*distance)/bb;
//        *(skernel+i) = 1;
        skernel.push_back(exp(aa*distance)/bb);
        sum += skernel[i];
//        cout<<*(kernel+i)<<"--"<<i<<"   ";
    }
//    cout<<"zhixingchulaila"<<endl;
//    cout<<"sum:"<<sum<<endl;

    for( i = 0;i<ksize;i++){
        skernel[i] = skernel[i]/sum;
//        cout<<skernel[i]<<"-*-"<<i<<" ";
    }
//    cout<<"执行出来拉！"<<endl;
    //cout<<"i:"<<i<<endl;


    //x方向的高斯模糊
    for(int y = 0;y < src.rows;y++){
        for(int x = 0;x < src.cols;x++){
            psum = 0;
            dvalue = 0;
            for( i = -kcenter;i <= kcenter;i++){
                if((x+i) >= 0 && (x+i) < src.cols){
                    dvalue += src.at<float>(y,x+i)*(skernel[kcenter+i]);
                    psum += skernel[kcenter+i];
                }

            }

            tmp.at<float>(y,x) = dvalue/psum;
        }
    }

//    cout << "x end"<<endl;
//    cout<<src.cols<<" - "<<src.rows<<endl;
    //y方向的高斯模糊
    for(int x = 0;x < src.cols;x++){
        for(int y = 0;y < src.rows;y++){
            psum = 0;
            dvalue = 0;
            for( i = -kcenter;i <= kcenter;i++){
                if((y+i) >= 0 && (y+i)<src.rows){
                    dvalue += tmp.at<float>(y+i,x)*(skernel[kcenter+i]);
                    psum += skernel[kcenter+i];
                }
//                cout<<"y ac:"<<dvalue<<endl;
            }

            dst.at<float>(y,x) = dvalue/psum;
//            cout<<"y inside:"<<y<<endl;
        }
    }
//    cout<<"y end"<<endl;

   // delete[] skernel;
   // skernel = NULL;
}

void CTlog(Mat &src,Mat &dst){
    int Height = src.cols;
    int Width = src.rows;
    int x,y;
    double dvalue;
    for(x = 0;x < Height;x++){
        for(y = 0;y < Width;y++){
            dvalue = log(src.at<uchar>(x,y));
            if(dvalue > 255){
                dst.at<uchar>(x,y) = 255;
            }
            if(dvalue<0){
                dst.at<uchar>(x,y) = 0;
            }
            else{
                dst.at<uchar>(x,y) = (int)dvalue;
            }
        }

    }
}

void subImg(Mat &src1,Mat &src2,Mat &dst){
    int Height = src1.cols;
    int Width = src1.rows;
    int x,y;
    int dvalue;
    for(x = 0;x < Height;x++){
        for(y = 0;y < Width;y++){
        //    dvalue  = src1.at<uchar>(x,y) - src2.at<uchar>(x,y);
            dst.at<uchar>(x,y) = src1.at<uchar>(x,y) - src2.at<uchar>(x,y);
            //这样以后得到的dst一片漆黑。。。
//            if(dvalue > 255){
//                dst.at<uchar>(x,y) = 255;
//            }
//            if(dvalue<0){
//                dst.at<uchar>(x,y) = 0;
//            }
//            else{
//                dst.at<uchar>(x,y) = dvalue;
//            }
        }
    }
}

void stretch(Mat &src){
    int Height = src.cols;
    int Width = src.rows;
    cout<<Height<<"--"<<Width<<endl;
    int x = 0,y = 0;
    double lmax=0,lmin = 255;
    double min,max,maxmin,dvalue,avg = 0,fc = 0;
    double dli;
    int scale = 2;
    for(x = 0;x < Height;x++){
        for(y = 0;y < Width;y++){
            dli = src.at<float>(y,x);
            if(dli > lmax){
                lmax = dli;
            }
            if(dli<lmin){
                lmin = dli;
            }
            avg += dli;
           // cout<<"dli:"<<dli<<"  ";
        }
    }
    cout<<endl;
    cout<<"lmax:"<<lmax<<endl;
    avg = avg/(Height*Width);
    cout<<"lmin:"<<lmin<<endl;
    dli = 0;
    for(x = 0;x < Height;x++){
        for(y = 0;y < Width;y++){
            dli = src.at<float>(y,x);
            fc = fc +(dli-avg)*(dli-avg);
        }
    }
    fc = fc/(Height*Width);
    cout<<"fc:-avg:"<<fc<<" - "<<avg<<endl;

    Scalar me,de;
    meanStdDev(src,me,de);
 //   cout<<"mmme"<<me.val[0]<<"---"<<de.val[0]<<endl;
    min = me.val[0] - scale*de.val[0];
    max = me.val[0] + scale*de.val[0];
  //  min = avg - scale*fc;
  //  max = avg + scale*fc;
    maxmin = max - min;
  //  cout<<"######"<<maxmin<<endl;
    for(x =0;x<Height;x++){
        for(y = 0;y < Width;y++){
             dvalue = ((src.at<float>(y,x)-min)/maxmin);
           // cout<<"***"<<dvalue<<endl;
             src.at<float>(y,x) = dvalue;
//            if(dvalue > 255){
//                src.at<float>(x,y) = 255;
//            }
//            if(dvalue<0){
//                src.at<float>(x,y) = 0;
//            }
//            else{
//                src.at<float>(x,y) = (int)dvalue;
//            }
        }
    }
}
void pImg(Mat &src){
    int Height = src.cols;
    int Width = src.rows;
    int x = 0,y = 0;
    for(x = 0;x < Height; x++){
        for(y = 0;y < Width;y++){
            cout<<src.at<float>(x,y)<<" ";
//            if(src.at<int>(x,y)>255||src.at<int>(x,y)<255){
//                cout<<"hhhhhcuolelelelelel"<<endl;
//                return;
//            }
        }
        cout<<endl;
    }
}

void addforlog(Mat &src,Mat &dst,float f){
    int Height = src.cols;
    int Width = src.rows;
    int x = 0,y = 0;
    for(x = 0;x < Height; x++){
        for(y = 0;y < Width;y++){
            dst.at<float>(y,x) = src.at<float>(y,x)+f;
        }
    }
}

void sunSSR(Mat &src,Mat &dst,double segma){
    Mat src_s,src_r;
    src_s.create(src.size(),CV_32F);
    dst.create(src.size(),CV_32F);
    cout<<"badd"<<endl;
    addforlog(src,dst,1.0);
    cout<<"a_add"<<endl;
//    convertScaleAbs(src,dst,1.0,1.0);

//    CTlog(dst,src_s);
    log(dst,src_s);
    //二维高斯模糊
//    int l = 1+2*ceil(3*segma);
//    float *Gkernel = new float(l*l);
//    Gkernel = makeGaussKern(segma);
//    Gauss2D(src,src_r,Gkernel,segma);
    //一维高斯模糊
    cout<<"bGauss"<<endl;
    Gauss(src,src_r,segma);

    cout<<"GAUSS"<<endl;
    //CTlog(src_r,src_r);
    addforlog(src_r,src_r,1.0);
    log(src_r,src_r);
    cout<<"substract"<<endl;
    //subImg(src_s,src_r,dst);
    subtract(src_s,src_r,dst);
    cout<<"stretch"<<endl;
    stretch(dst);
    cout<<"after stretch"<<endl;

    return;

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
    //执行retinex时用30比较好，实验时先写5
    double segma = 30;
    int l = 1+2*ceil(3*segma);
    string filename="/Users/sunguoyan/Downloads/picture/lena.bmp";

    Mat src,src1,dst;
    src = imread(filename,CV_LOAD_IMAGE_GRAYSCALE);
   // src1.create(src.size(),src.type());
    src.convertTo(src1,CV_32F,1/255.0);
    //二维高斯模糊
//    float *Gkernel = new float(l*l);
//    Gkernel = makeGaussKern(segma);
//    Gauss2D(src1,dst,Gkernel,segma);

  //  Gauss(src1,dst,segma);
  //  log(src1,dst);
    sunSSR(src1,dst,segma);
    namedWindow("test");
    namedWindow("test1");
    imshow("test",src1);
    imshow("test1",dst);
    waitKey(0);

    cout<<"到return前拉"<<endl;
    return 0;
}