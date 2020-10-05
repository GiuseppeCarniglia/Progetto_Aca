#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <math.h>
#include <omp.h>

using namespace cv;
using namespace std;

#define ZERO_VALUE (ushort)0



/*
________________________________________________Haar Transform_________________________________________________________
*/

Mat Haar_antitrasformata(Mat immagine_trasformata);


int main() {
	const char* imgName = "./immagini/leonessa.jpg";
	Mat image = imread(imgName, IMREAD_GRAYSCALE);
	
	Mat dst = cv::Mat(image.rows,image.cols,CV_8UC1);
	int imageWidth = image.size().width;
	int w = 0;
	int k = 0;
	
	double initial_time = 0, final_time = 0;
	int temp_row = 0;
	int temp_col = 0;

	int i=0;
	int j=0;
	
	temp_row = (image.rows);
	temp_col = (image.cols);
	
	w = (image.cols);
	k = (image.rows);
	
	initial_time = omp_get_wtime();
	
	//immagine A e D			
	omp_set_num_threads(omp_get_max_threads());
	
	#pragma omp parallel for private(i,j) schedule(static)
	for (i = 0; i < temp_row; i++) 
	{
		for (int j = 0; j < w/2; j++) {
			dst.at<uchar>(i, j) = (image.at<uchar>(i,2*j) + image.at<uchar>(i,2*j + 1)) / 2;
			
			dst.at<uchar>(i, j+(w/2)) = (image.at<uchar>(i,2*j) - image.at<uchar>(i,2*j+1))/2;
		}
		
		printf("N Threads. %d, Thread ID %d\n",omp_get_num_threads(), omp_get_thread_num());
	}
			
	#pragma omp for private(i,j) schedule(static)
	for (int i = 0; i < temp_row; i++) {
		for (int j = 0; j < temp_col; j++) {
			image.at<uchar>(i, j) = dst.at<uchar>(i, j);
		}
	}
	
	#pragma omp parallel for private(i,j) schedule(static)
	for (int i = 0; i < w; i++) {		
		for (int j = 0; j < k/2; j++) {
			dst.at<uchar>(j, i) = (image.at<uchar>(2*j,i) + image.at<uchar>(2*j + 1, i)) / 2;
			dst.at<uchar>(j+(k/2), i) = (image.at<uchar>(2*j,i) - image.at<uchar>(2*j+1,i))/2;
		}
	}
	
	#pragma omp for private(i,j) schedule(static)
	for (int i = 0; i < w; i++) {
		for (int j = 0; j < k; j++) {
			image.at<uchar>(j, i) = dst.at<uchar>(j,i);
		}
	}
	
	final_time = omp_get_wtime();
	final_time -= initial_time;
	printf("time: %lf \n", final_time);

	vector<int> compression_params;

	compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);

	compression_params.push_back(60);

	imwrite("./immagini_modificate/Haar_parallela/Haar_parallela.jpg",image,compression_params);
	imwrite("./immagini_modificate/Haar_parallela/Haar_parallela_antitrasformata.jpg",Haar_antitrasformata(image),compression_params);
	
	waitKey(0);
	return 0;
}


Mat Haar_antitrasformata(Mat immagine_trasformata){
	
	Mat image_out = cv::Mat(immagine_trasformata.rows, immagine_trasformata.cols, CV_8UC1);
	
	Mat image_out_2 = cv::Mat(immagine_trasformata.rows, immagine_trasformata.cols, CV_8UC1);

	
	int k = immagine_trasformata.rows;
	int w = immagine_trasformata.cols;
 	int i,j;	

	#pragma omp parallel for private(i,j) schedule(static)
	for(i=0;i<k/2;i++){
		for(j=0;j<w;j++){
			image_out.at<uchar>(2*i,j) = immagine_trasformata.at<uchar>(i,j) + immagine_trasformata.at<uchar>(i+(k/2),j);
			
			image_out.at<uchar>(2*i+1,j) = immagine_trasformata.at<uchar>(i,j) -  immagine_trasformata.at<uchar>(i+(k/2),j);
			
		}
	}
	
	#pragma omp parallel for private(i,j) schedule(static)
	for(i=0;i<k;i++){
		for(j=0;j<w/2;j++){
			image_out_2.at<uchar>(i,2*j) = image_out.at<uchar>(i,j) + image_out.at<uchar>(i,j+(w/2));
			
			image_out_2.at<uchar>(i,2*j+1) = image_out.at<uchar>(i,j) -  image_out.at<uchar>(i,j+(w/2));
			
		}
	}
	return image_out_2;
	
}





