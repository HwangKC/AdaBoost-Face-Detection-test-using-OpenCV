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
		//모든 영역 그리기 //draw all rect
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
		gpu_faceBuf.colRange(0, detectionNumber).download(faces_downloaded); //0~detectionNumber column값 Mat으로 다운로드
		Rect* faces = faces_downloaded.ptr< Rect>();
		//모든 영역 그리기 //draw all rectangle
		for(int i = 0; i < detectionNumber; ++i)
		{
			rectangle(img, Point(faces[i].x, faces[i].y), Point(faces[i].x+faces[i].width, faces[i].y+faces[i].height), CV_RGB(255,0,0), 2);
		}
	}
}


void main()
{
	//계산 시간 측정을 위해 //for estimation porcessing time
	float TakeTime;
	unsigned long Atime, Btime;

	//결과 창
	namedWindow("Result Window", CV_WINDOW_FREERATIO);

	//이미지 읽기
	Mat img = imread("afterschool.jpg");
	Mat grayImg; //AdaBoost는 회색영상만 가능함 // AdaBoost possible only grayscale
	cvtColor(img, grayImg, CV_BGR2GRAY); //회색영상 만듦 //make grayscale

	//xml 학습 데이터 읽기 //load xml learned data
	//string trainface = ".\\cascade_torso_LBP.xml";
	string trainface = ".\\cascade_LBP3.xml";

	//cpu, gpu 버전 adaboost 클래스 만들기 //make adaboost class for cpu, gpu version
	CascadeClassifier ada_cpu;
	gpu::CascadeClassifier_GPU ada_gpu;

	//cpu버전 adaboost xml 데이터 읽기 //load adaboost xml data
	if(!(ada_cpu.load(trainface)))
	{
		printf("cpu-adaboost xml 파일 읽기 실패\n"); //loading fail
		return ;
	}

	//gpu버전 adaboost xml 데이터 읽기
	if(!(ada_gpu.load(trainface)))
	{
		printf("gpu-adaboost xml 파일 읽기 실패\n"); //loading fail
		return ;
	}

	//cpu 버전 adaboost 얼굴 검출 //cpu version, adaboost for detecting face
	vector< Rect > faces; //검출 결과를 위해 //for result out
	Atime = getTickCount(); //시작 시간 //start time
	ada_cpu.detectMultiScale(grayImg, faces, 1.2, 3, 4, Size(64,64));
	ProccTimePrint( Atime , "cpu AdaBoost :"); //처리시간 출력 //print processing time
	printf("CPU %d faces :BLUE\n", faces.size());
	CpuAdaResultDraw(faces, img); //얼굴 영역 그리기 //draw face rect

	//gpu 버전 adaboost 얼굴 검출 //gpu version adaboost for detecting face
	gpu::GpuMat gpu_faceBuf; //검출 결과 저장 //output gpu result
	gpu::GpuMat gpu_Img;
	gpu_Img.upload(grayImg); //GpuMat으로 업로드 //upload 
	Atime = getTickCount(); //시작 시간 //processing start
	int detectionNumber = ada_gpu.detectMultiScale(gpu_Img, gpu_faceBuf, 1.2, 4, Size(64,64) );  
	ProccTimePrint( Atime , "gpu AdaBoost :"); //처리시간 출력 //print processing time
	printf("GPU %d faces :RED\n", detectionNumber);
	GpuAdaResultDraw(detectionNumber, gpu_faceBuf, img); //얼굴 영역 그리기 //draw face rectangle

	//결과 출력 //image output
	imshow("Result Window", img);
	waitKey(0);

}