#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

#define ZERO_VALUE (ushort)0



/*
________________________________________________Haar Transform_________________________________________________________
*/
#include<omp.h>

Mat Haar_antitrasformata(Mat immagine_trasformata);

Mat diff_of_images(Mat original, Mat antitransform);


int main() {
	const char* imgName = "./immagini/leonessa.jpg";
	Mat image_original = imread(imgName, IMREAD_GRAYSCALE);
	Mat image = imread(imgName, IMREAD_GRAYSCALE);

	
	Mat dst = cv::Mat(image.rows,image.cols,CV_8UC1);

	Mat antitransform = cv::Mat(image.rows,image.cols,CV_8UC1);
	Mat image_difference = cv::Mat(image.rows,image.cols,CV_8UC1);
	int imageWidth = image.size().width;
	int w = 0;
	int k = 0;
	uchar* a;
	a = (uchar*)malloc(image.cols * sizeof(uchar));
	
	uchar* d;
	d = (uchar*)malloc(image.cols * sizeof(uchar));
	int iteration = 0; //numero deell'iterazione corrente
	int max_iteration = 2;//massimo numero di iterazioni, se vuoi farglielo fare più di una volta aumenta questo valore
	double initial_time = 0, final_time = 0;
	int temp_row = 0;
	int temp_col = 0;

	initial_time = omp_get_wtime();
	
	int i=0;
	int j=0;
	
	temp_row = (image.rows / pow(2, iteration));
	temp_col = (image.cols / pow(2, iteration));
	
			//immagine A e D
			for (i = 0; i < temp_row; i++) 
			{
			w = (image.cols / pow(2, iteration));

				for (j = 0; j < w / 2; j++) {
					dst.at<uchar>(i,j) = (image.at<uchar>(i, j + j) + image.at<uchar>(i, j + j + 1)) / 2;
					
				}
				for (j = 0; j < w/2; j++) {
					dst.at<uchar>(i,j+(w/2)) = (image.at<uchar>(i, j +j) - image.at<uchar>(i, j+j+1))/2;
				}
				
			}
				
	
		for (int i = 0; i < temp_row; i++) {
			for (int j = 0; j < temp_col; j++) {
				image.at<uchar>(i, j) = dst.at<uchar>(i, j);
			}
		}
		


	for (int i = 0; i < temp_col; i++) {
		w = (image.rows / pow(2, iteration));
		k = 0;
		
		for (int j = 0; j < w / 2; j++) {
			dst.at<uchar>(j,i) = (image.at<uchar>(2*j, i) + image.at<uchar>(2*j+1, i)) / 2;

			dst.at<uchar>(j+(w/2),i) = (image.at<uchar>(2*j, i) - image.at<uchar>(2*j+1, i)) / 2;
		}
		
		for (int j = 0; j < temp_row; j++) {
			image.at<uchar>(j, i) = dst.at<uchar>(j,i);
		}
	}
	final_time = omp_get_wtime();
	final_time -= initial_time;
	free(a);
	free(d);
	printf("time: %lf \n", final_time);
	

	antitransform = Haar_antitrasformata(image);
	image_difference = diff_of_images(image_original, antitransform);

	double min, max;
	cv::minMaxLoc(image_difference, &min, &max);

	printf("Image difference\nLowest value:%lf Highest value:%lf\n",min,max);

	vector<int> compression_params;

	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);

	compression_params.push_back(60);

	imwrite("./immagini_modificate/Haar_seriale/Haar_seriale.jpg",image,compression_params);
	
	imwrite("./immagini_modificate/Haar_seriale/Haar_seriale_antitrasformata.jpg",antitransform,compression_params);
	imwrite("./immagini_modificate/Haar_seriale/Haar_seriale_difference.jpg",image_difference,compression_params);

	waitKey(0);
	
	return 0;
}


Mat Haar_antitrasformata(Mat immagine_trasformata){
	
	Mat image_out = cv::Mat(immagine_trasformata.rows, immagine_trasformata.cols, CV_8UC1);
	Mat image_out_2 = cv::Mat(immagine_trasformata.rows, immagine_trasformata.cols, CV_8UC1);

	int k = immagine_trasformata.rows;
	int w = immagine_trasformata.cols;
	int i=0, j=0;	
	for(i=0;i<k/2;i++){
		for(j=0;j<w;j++){
			image_out.at<uchar>(2*i,j) = immagine_trasformata.at<uchar>(i,j) + immagine_trasformata.at<uchar>(i+(k/2),j);
			
			image_out.at<uchar>(2*i+1,j) = immagine_trasformata.at<uchar>(i,j) - immagine_trasformata.at<uchar>(i+(k/2),j);
			
		}
	}


	
	for(i=0;i<k;i++){
		for(j=0;j<w/2;j++){
			image_out_2.at<uchar>(i,2*j) = image_out.at<uchar>(i,j) + image_out.at<uchar>(i,j+(w/2));
			
			image_out_2.at<uchar>(i,2*j+1) = image_out.at<uchar>(i,j) - image_out.at<uchar>(i,j+(w/2));
			
		}
	}
	
	return image_out_2;
	
	}


Mat diff_of_images(Mat original, Mat antitransform){

	if(original.cols != antitransform.cols || original.rows != antitransform.rows){
		perror("Images of different sizes");

	}

	Mat image_diff = cv::Mat(original.rows, original.cols, CV_8UC1);
	
	int i=0, j=0;

	for(i=0;i<original.rows;i++){

		for(j=0;j<original.cols;j++){

			image_diff.at<uchar>(i,j) = original.at<uchar>(i,j) - antitransform.at<uchar>(i,j);
		}
	}

	return image_diff;
}

