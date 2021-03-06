#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <math.h>
#include<omp.h>
#include<stdlib.h>

using namespace cv;
using namespace std;

#define ZERO_VALUE (double)0



/*
________________________________________________Haar Transform_________________________________________________________
*/

Mat Haar_antitrasformata(Mat immagine_trasformata, int nOfLevels);


Mat diff_of_images(Mat original, Mat antitransform);


int main(int argc, char *argv[]) {
	const char* imgName = argv[1];
	Mat image_original = imread(imgName, IMREAD_GRAYSCALE);
	Mat image = imread(imgName, IMREAD_GRAYSCALE);
	
	image.convertTo(image, CV_64FC1);
	
	int imageHeight = image.rows;
	int imageWidth = image.cols;

	Mat dst = cv::Mat(image.rows,image.cols,CV_64FC1);

	Mat antitransform = cv::Mat(image.rows,image.cols,CV_64FC1);
	Mat image_difference = cv::Mat(image.rows,image.cols,CV_64FC1);
		
	int w = 0;

	double initial_time = 0, final_time = 0;
	int temp_row = 0;
	int temp_col = 0;


	int i=0;
	int j=0;
	int k=0;
	int nOfLevels = atoi(argv[2]);
	
	if(nOfLevels <= 0){
	
		perror("Numero di livelli deve essere >= 0\n");
		return 1;
	}
	
	

	temp_row = (image.rows);
	temp_col = (image.cols);

	w = imageWidth;


	initial_time = omp_get_wtime();

	//immagine A e D
	for(k=0;k<nOfLevels;k++){

    	for (i = 0; i < imageHeight; i++){
    		for (j = 0; j < imageWidth / 2; j++) {
    			dst.at<double>(i,j) = (image.at<double>(i,2*j) + image.at<double>(i,2*j + 1)) / 2;
			
    			dst.at<double>(i,j+(w/2)) = (image.at<double>(i,2*j) - image.at<double>(i,2*j+1))/2;
    		}
    	}

	    w = imageHeight;

		for (int i = 0; i < imageHeight / 2; i++) {

			for(int j=0;j<imageWidth;j++){
				image.at<double>(i,j) = (dst.at<double>(2*i, j) + dst.at<double>(2*i+1, j)) / 2;

				image.at<double>(i+(w/2),j) = (dst.at<double>(2*i, j) - dst.at<double>(2*i+1, j)) / 2;
				}
		}
		
		for (int i = 0; i < imageHeight; i++) {
			for (int j = 0; j < imageWidth; j++) {
				dst.at<double>(i, j) = image.at<double>(i, j);
			}
		}
		
		
		imageHeight = imageHeight / 2;
		imageWidth = imageWidth / 2;
		
		w = imageWidth;
		
		
    }
        
	final_time = omp_get_wtime();
	final_time -= initial_time;
	printf("%lf\n", final_time);

/*	
	cv::Mat final_image(imageHeight,imageWidth,CV_8UC1);
	
	cv::normalize(image,final_image, 0,255,NORM_MINMAX,CV_8UC1);
	
	imshow("Haar transform serial",final_image);
		
	antitransform = Haar_antitrasformata(image,nOfLevels);
	
	cv::normalize(antitransform,final_image, 0,255,NORM_MINMAX,CV_8UC1);
	
	imshow("Haar antitransform serial",final_image);

	image_difference = diff_of_images(image, antitransform);

	cv::normalize(image_difference,final_image, 0,255,NORM_MINMAX,CV_8UC1);
	
	imshow("Haar difference",final_image);
*/
//	double min, max;
//	cv::minMaxLoc(image_difference, &min, &max);

//	printf("Image difference\nLowest value:%lf Highest value:%lf\n",min,max);

	waitKey(0);

	return 0;
}

// Haar antitransform
Mat Haar_antitrasformata(Mat immagine_trasformata, int nOfLevels){

	Mat image_out;
	Mat image_out_2;

	int k = immagine_trasformata.rows/pow(2, nOfLevels - 1);
	int w = immagine_trasformata.cols/pow(2, nOfLevels - 1);
	int i=0, j=0, t=0;



	for(t=0;t<nOfLevels;t++){
		image_out = cv::Mat(k,w, CV_64FC1);
		image_out_2 = cv::Mat(k,w, CV_64FC1);
	
	for(i=0;i<k/2;i++){
		for(j=0;j<w;j++){
			image_out.at<double>(2*i,j) = immagine_trasformata.at<double>(i,j) + immagine_trasformata.at<double>(i+(k/2),j);

			image_out.at<double>(2*i+1,j) = immagine_trasformata.at<double>(i,j) - immagine_trasformata.at<double>(i+(k/2),j);

		}
	}

	for(i=0;i<k;i++){
		for(j=0;j<w/2;j++){
			image_out_2.at<double>(i,2*j) = image_out.at<double>(i,j) + image_out.at<double>(i,j+(w/2));

			image_out_2.at<double>(i,2*j+1) = image_out.at<double>(i,j) - image_out.at<double>(i,j+(w/2));

		}
	}
	
	for(i=0;i<k;i++){
		for(j=0;j<w;j++){
		
			immagine_trasformata.at<double>(i,j) = image_out_2.at<double>(i,j);
		
		}
	}
	
		k = 2*k;
		w = 2*w;
	}

	return immagine_trasformata;
}

//Difference of images
Mat diff_of_images(Mat original, Mat antitransform){

	if(original.cols != antitransform.cols || original.rows != antitransform.rows){
		perror("Images of different sizes");
	}

	Mat image_diff = cv::Mat(original.rows, original.cols, CV_64FC1);

	int i=0, j=0;

	for(i=0;i<original.rows;i++){

		for(j=0;j<original.cols;j++){

			image_diff.at<double>(i,j) = original.at<double>(i,j) - antitransform.at<double>(i,j);
		}
	}

	return image_diff;
}
