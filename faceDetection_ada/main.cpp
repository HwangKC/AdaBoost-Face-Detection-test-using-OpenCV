#include < stdio.h>   
#include < opencv2\opencv.hpp>   
#include < opencv2\gpu\gpu.hpp>   
#include < vector>

using namespace cv;
using namespace std;     

#ifdef _DEBUG           
#pragma comment(lib, "opencv_core249d.lib")            
#pragma comment(lib, "opencv_imgproc249d.lib")
#pragma comment(lib, "opencv_objdetect249d.lib")
#pragma comment(lib, "opencv_gpu249d.lib")           
#pragma comment(lib, "opencv_highgui249d.lib")                
#else           
#pragma comment(lib, "opencv_core249.lib")           
#pragma comment(lib, "opencv_imgproc249.lib")           
#pragma comment(lib, "opencv_objdetect249.lib")           
#pragma comment(lib, "opencv_gpu249.lib")           
#pragma comment(lib, "opencv_highgui249.lib")           
#endif    

void ProccTimePrint( unsigned long Atime , string msg)
{
	unsigned long Btime=0;
	float sec, fps;
	Btime = getTickCount();
	sec = (Btime - Atime)/getTickFrequency();
	fps = 1/sec;
	printf("%s %.4lf(sec) / %.4lf(fps) \n", msg.c_str(),  sec, fps );
}

void CpuAdaResultDraw(vector< Rect > &faces, Mat& img)
{
	if( faces.size() >=1 )
	{
		//��� ���� �׸��� //draw all rect
		for(int i = 0; i < faces.size(); ++i)
		{
			rectangle(img, faces[i], CV_RGB(0,0,255), 4);
		}
	}
}

void GpuAdaResultDraw(int detectionNumber, gpu::GpuMat &gpu_faceBuf, Mat& img)
{
	if(detectionNumber >= 1)
	{
		Mat faces_downloaded; 
		gpu_faceBuf.colRange(0, detectionNumber).download(faces_downloaded); //0~detectionNumber column�� Mat���� �ٿ�ε�
		Rect* faces = faces_downloaded.ptr< Rect>();
		//��� ���� �׸��� //draw all rectangle
		for(int i = 0; i < detectionNumber; ++i)
		{
			rectangle(img, Point(faces[i].x, faces[i].y), Point(faces[i].x+faces[i].width, faces[i].y+faces[i].height), CV_RGB(255,0,0), 2);
		}
	}
}


void main()
{
	//��� �ð� ������ ���� //for estimation porcessing time
	float TakeTime;
	unsigned long Atime, Btime;

	//��� â
	namedWindow("Result Window", CV_WINDOW_FREERATIO);

	//�̹��� �б�
	Mat img = imread("afterschool.jpg");
	Mat grayImg; //AdaBoost�� ȸ������ ������ // AdaBoost possible only grayscale
	cvtColor(img, grayImg, CV_BGR2GRAY); //ȸ������ ���� //make grayscale

	//xml �н� ������ �б� //load xml learned data
	//string trainface = ".\\cascade_torso_LBP.xml";
	string trainface = ".\\cascade_LBP3.xml";

	//cpu, gpu ���� adaboost Ŭ���� ����� //make adaboost class for cpu, gpu version
	CascadeClassifier ada_cpu;
	gpu::CascadeClassifier_GPU ada_gpu;

	//cpu���� adaboost xml ������ �б� //load adaboost xml data
	if(!(ada_cpu.load(trainface)))
	{
		printf("cpu-adaboost xml ���� �б� ����\n"); //loading fail
		return ;
	}

	//gpu���� adaboost xml ������ �б�
	if(!(ada_gpu.load(trainface)))
	{
		printf("gpu-adaboost xml ���� �б� ����\n"); //loading fail
		return ;
	}

	//cpu ���� adaboost �� ���� //cpu version, adaboost for detecting face
	vector< Rect > faces; //���� ����� ���� //for result out
	Atime = getTickCount(); //���� �ð� //start time
	ada_cpu.detectMultiScale(grayImg, faces, 1.2, 3, 4, Size(64,64));
	ProccTimePrint( Atime , "cpu AdaBoost :"); //ó���ð� ��� //print processing time
	printf("CPU %d faces :BLUE\n", faces.size());
	CpuAdaResultDraw(faces, img); //�� ���� �׸��� //draw face rect

	//gpu ���� adaboost �� ���� //gpu version adaboost for detecting face
	gpu::GpuMat gpu_faceBuf; //���� ��� ���� //output gpu result
	gpu::GpuMat gpu_Img;
	gpu_Img.upload(grayImg); //GpuMat���� ���ε� //upload 
	Atime = getTickCount(); //���� �ð� //processing start
	int detectionNumber = ada_gpu.detectMultiScale(gpu_Img, gpu_faceBuf, 1.2, 4, Size(64,64) );  
	ProccTimePrint( Atime , "gpu AdaBoost :"); //ó���ð� ��� //print processing time
	printf("GPU %d faces :RED\n", detectionNumber);
	GpuAdaResultDraw(detectionNumber, gpu_faceBuf, img); //�� ���� �׸��� //draw face rectangle

	//��� ��� //image output
	imshow("Result Window", img);
	waitKey(0);

}