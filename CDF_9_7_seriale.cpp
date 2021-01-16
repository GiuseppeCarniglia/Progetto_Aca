#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <math.h>
#include <omp.h>

using namespace cv;
using namespace std;

#define ZERO_VALUE (uchar)0

/*
   
   
   ________________________________________________9/7-CDF Transform_________________________________________________________
   * 
   * 
  
 */


int main(int argc, char *argv[])	
{
	const char* imgName = argv[1];
	Mat image = imread(imgName, IMREAD_GRAYSCALE);
	int imageWidth = image.size().width;
	int imageHeight = image.size().height;
	Mat new_image = cv::Mat(imageHeight,imageWidth,CV_8UC1);

	int w = imageWidth;

	double coefficientsLP[5] = { 0.602949018236,0.266864118443,-0.078223266529,-0.016864118443,0.26748757411 };
	double coefficientsHP[5] = { 1.11508705, -0.59127176314, -0.057543526229, 0.091271763114, 0.0 };
	int padding[3] = { (w / 2) - 2 ,(w / 2) - 1 ,(w / 2) };
	double initial_time = 0, final_time = 0;
	int i=0,j=0;

	double time_A=0,time_D=0,time_AA=0,time_DD=0;
	initial_time = omp_get_wtime();

	
	time_A = omp_get_wtime();
	for (i = 0; i < imageHeight; i++) {
	
		//Calcolo matrice di approssimazione a
		new_image.at<uchar>(i,0) = coefficientsLP[0] * image.at<uchar>(i,0) 
			+ coefficientsLP[1] * (image.at<uchar>(i, 1) + ZERO_VALUE)
			+ coefficientsLP[2] * (image.at<uchar>(i, 2) + ZERO_VALUE)
			+ coefficientsLP[3] * (image.at<uchar>(i, 3) + ZERO_VALUE)
			+ coefficientsLP[4] * (image.at<uchar>(i, 4) + ZERO_VALUE);

		new_image.at<uchar>(i,1) = coefficientsLP[0] * image.at<uchar>(i, 2) 
			+ coefficientsLP[1] * (image.at<uchar>(i, 3) + image.at<uchar>(i, 1))
			+ coefficientsLP[2] * (image.at<uchar>(i, 4) + image.at<uchar>(i, 0))
			+ coefficientsLP[3] * (image.at<uchar>(i, 5) + ZERO_VALUE)
			+ coefficientsLP[4] * (image.at<uchar>(i, 6) + ZERO_VALUE);

		new_image.at<uchar>(i,2) = coefficientsLP[0] * image.at<uchar>(i, 4) 
			+ coefficientsLP[1] * (image.at<uchar>(i, 5) + image.at<uchar>(i, 3))
			+ coefficientsLP[2] * (image.at<uchar>(i, 6) + image.at<uchar>(i, 2))
			+ coefficientsLP[3] * (image.at<uchar>(i, 7) + image.at<uchar>(i, 1))
			+ coefficientsLP[4] * (image.at<uchar>(i, 8) + image.at<uchar>(i, 0));


		for (j = 3; j < (w/2) - 2; j++) {
			new_image.at<uchar>(i,j) = coefficientsLP[0] * image.at<uchar>(i, (j<<1))
				+ coefficientsLP[1] * (image.at<uchar>(i, (j<<1) + 1) + image.at<uchar>(i, (j<<1) - 1))
				+ coefficientsLP[2] * (image.at<uchar>(i, (j<<1) + 2) + image.at<uchar>(i, (j<<1) - 2))
				+ coefficientsLP[3] * (image.at<uchar>(i, (j<<1) + 3) + image.at<uchar>(i, (j<<1) - 3))
				+ coefficientsLP[4] * (image.at<uchar>(i, (j<<1) + 4) + image.at<uchar>(i, (j<<1) - 4));
		}

		new_image.at<uchar>(i,padding[0]) = coefficientsLP[0] * image.at<uchar>(i,(padding[0]<<1))
			+ coefficientsLP[1] * (image.at<uchar>(i, (padding[0]<<1) + 1) 
			+ image.at<uchar>(i, (padding[0]<<1) - 1))
			+ coefficientsLP[2] * (image.at<uchar>(i, (padding[0]<<1) + 2) 
			+ image.at<uchar>(i, (padding[0]<<1) - 2))
			+ coefficientsLP[3] * (image.at<uchar>(i, (padding[0]<<1) + 3) 
			+ image.at<uchar>(i, (padding[0]<<1) - 3))
			+ coefficientsLP[4] * (ZERO_VALUE + image.at<uchar>(i, (padding[0]<<1) - 4));

		new_image.at<uchar>(i,padding[1]) = coefficientsLP[0] * image.at<uchar>(i, (padding[1]<<1))
			+ coefficientsLP[1] * (image.at<uchar>(i,(padding[1]<<1) + 1) 
			+ image.at<uchar>(i, (padding[1]<<1) - 1))
			+ coefficientsLP[2] * (ZERO_VALUE + image.at<uchar>(i, (padding[1]<<1) - 2))
			+ coefficientsLP[3] * (ZERO_VALUE + image.at<uchar>(i, (padding[1]<<1) - 3))
			+ coefficientsLP[4] * (ZERO_VALUE + image.at<uchar>(i, (padding[1]<<1) - 4));

		new_image.at<uchar>(i,padding[2]) = ZERO_VALUE
			+ coefficientsLP[1] * (ZERO_VALUE + image.at<uchar>(i, (padding[2] << 1) - 1))
			+ coefficientsLP[2] * (ZERO_VALUE + image.at<uchar>(i, (padding[2] << 1) - 2))
			+ coefficientsLP[3] * (ZERO_VALUE + image.at<uchar>(i, (padding[2] << 1) - 3))
			+ coefficientsLP[4] * (ZERO_VALUE + image.at<uchar>(i, (padding[2] << 1) - 4));
	}

	for (i = 0; i < image.rows ; i++) {
		//Calcolo matrice di dettaglio d

		new_image.at<uchar>(i,0+(w/2)) = coefficientsHP[0] * image.at<uchar>(i, 0)
			+ coefficientsHP[1] * (image.at<uchar>(i, 1) + ZERO_VALUE)
			+ coefficientsHP[2] * (image.at<uchar>(i, 2) + ZERO_VALUE)
			+ coefficientsHP[3] * (image.at<uchar>(i, 3) + ZERO_VALUE)
			+ coefficientsHP[4] * (image.at<uchar>(i, 4) + ZERO_VALUE);

		new_image.at<uchar>(i,1+(w/2)) = coefficientsHP[0] * image.at<uchar>(i, 2)
			+ coefficientsHP[1] * (image.at<uchar>(i, 3) + image.at<uchar>(i, 1))
			+ coefficientsHP[2] * (image.at<uchar>(i, 4) + image.at<uchar>(i, 0))
			+ coefficientsHP[3] * (image.at<uchar>(i, 5) + ZERO_VALUE)
			+ coefficientsHP[4] * (image.at<uchar>(i, 6) + ZERO_VALUE);

		new_image.at<uchar>(i,2+(w/2)) = coefficientsHP[0] * image.at<uchar>(i, 4)
			+ coefficientsHP[1] * (image.at<uchar>(i, 5) + image.at<uchar>(i, 3))
			+ coefficientsHP[2] * (image.at<uchar>(i, 6) + image.at<uchar>(i, 2))
			+ coefficientsHP[3] * (image.at<uchar>(i, 7) + image.at<uchar>(i, 1))
			+ coefficientsHP[4] * (image.at<uchar>(i, 8) + image.at<uchar>(i, 0));

		
		for (j = 3; j < (imageWidth/2) - 2; j++) {
			 new_image.at<uchar>(i,j+(w/2)) = coefficientsHP[0] * image.at<uchar>(i, (j<<1))
				+ coefficientsHP[1] * (image.at<uchar>(i, (j<<1) + 1)+image.at<uchar>(i, (j<<1) - 1))
				+ coefficientsHP[2] * (image.at<uchar>(i, (j<<1) + 2)+image.at<uchar>(i, (j<<1) - 2))
				+ coefficientsHP[3] * (image.at<uchar>(i, (j<<1) + 3)+image.at<uchar>(i, (j<<1) - 3))
				+ coefficientsHP[4] * (image.at<uchar>(i, (j<<1) + 4)+image.at<uchar>(i, (j<<1) - 4));
		}
		
		new_image.at<uchar>(i,padding[0]+(w/2)) = coefficientsHP[0] * image.at<uchar>(i, (padding[0] << 1))
			+ coefficientsHP[1] * (image.at<uchar>(i, (padding[0] << 1) + 1) 
				+ image.at<uchar>(i, (padding[0] << 1) - 1))
			+ coefficientsHP[2] * (image.at<uchar>(i, (padding[0] << 1) + 2) 
				+ image.at<uchar>(i, (padding[0] << 1) - 2))
			+ coefficientsHP[3] * (image.at<uchar>(i, (padding[0] << 1) + 3) 
				+ image.at<uchar>(i, (padding[0] << 1) - 3))
			+ coefficientsHP[4] * (ZERO_VALUE + image.at<uchar>(i, (padding[0] << 1) - 4));
		
		new_image.at<uchar>(i,padding[1]+(w/2)) = coefficientsHP[0] * image.at<uchar>(i, (padding[1] << 1))
			+ coefficientsHP[1] * (image.at<uchar>(i, (padding[1]<<1)+1) + image.at<uchar>(i, (padding[1] << 1) - 1))
			+ coefficientsHP[2] * (ZERO_VALUE + image.at<uchar>(i, (padding[1] << 1) - 2))
			+ coefficientsHP[3] * (ZERO_VALUE + image.at<uchar>(i, (padding[1] << 1) - 3))
			+ coefficientsHP[4] * (ZERO_VALUE + image.at<uchar>(i, (padding[1] << 1) - 4));
		
		new_image.at<uchar>(i,padding[2]+(w/2)) = ZERO_VALUE 
			+ coefficientsHP[1] * (ZERO_VALUE + image.at<uchar>(i, (padding[2]<<1) - 1))
			+ coefficientsHP[2] * (ZERO_VALUE + image.at<uchar>(i, (padding[2]<<1) - 2))
			+ coefficientsHP[3] * (ZERO_VALUE + image.at<uchar>(i, (padding[2]<<1) - 3))
			+ coefficientsHP[4] * (ZERO_VALUE + image.at<uchar>(i, (padding[2]<<1) - 4));
	}

	time_D = omp_get_wtime();
	time_D -= time_A;



	image = new_image.clone();

	w = imageHeight;
	padding[0] = (w/2)-2;
	padding[1] = (w/2)-1;
	padding[2] = (w/2);
	
	time_AA =  omp_get_wtime();
	
	for (i = 0; i < imageWidth; i++) {
		
		//Calcolo matrice di approssimazione aa e da
		new_image.at<uchar>(0, i) = coefficientsLP[0] * image.at<uchar>(0,i) 
			+ coefficientsLP[1] * (image.at<uchar>(1,i) + ZERO_VALUE)
			+ coefficientsLP[2] * (image.at<uchar>(2,i) + ZERO_VALUE)
			+ coefficientsLP[3] * (image.at<uchar>(3,i) + ZERO_VALUE)
			+ coefficientsLP[4] * (image.at<uchar>(4,i) + ZERO_VALUE);

		new_image.at<uchar>(1, i) = coefficientsLP[0] * image.at<uchar>(2,i) 
			+ coefficientsLP[1] * (image.at<uchar>(3,i) + image.at<uchar>(1,i))
			+ coefficientsLP[2] * (image.at<uchar>(4,i) + image.at<uchar>(0,i))
			+ coefficientsLP[3] * (image.at<uchar>(5,i) + ZERO_VALUE)
			+ coefficientsLP[4] * (image.at<uchar>(6,i) + ZERO_VALUE);

		new_image.at<uchar>(2, i) = coefficientsLP[0] * image.at<uchar>(4,i) 
			+ coefficientsLP[1] * (image.at<uchar>(5,i) + image.at<uchar>(3,i))
			+ coefficientsLP[2] * (image.at<uchar>(6,i) + image.at<uchar>(2,i))
			+ coefficientsLP[3] * (image.at<uchar>(7,i) + image.at<uchar>(1,i))
			+ coefficientsLP[4] * (image.at<uchar>(8,i) + image.at<uchar>(0,i));


		for (j = 3; j < (w/2) - 2; j++) {
			new_image.at<uchar>(j, i) = coefficientsLP[0] * image.at<uchar>((j<<1),i)
				+ coefficientsLP[1] * (image.at<uchar>((j<<1) + 1,i) + image.at<uchar>((j<<1) - 1,i))
				+ coefficientsLP[2] * (image.at<uchar>((j<<1) + 2,i) + image.at<uchar>((j<<1) - 2,i))
				+ coefficientsLP[3] * (image.at<uchar>((j<<1) + 3,i) + image.at<uchar>((j<<1) - 3,i))
				+ coefficientsLP[4] * (image.at<uchar>((j<<1) + 4,i) + image.at<uchar>((j<<1) - 4,i));
		}

		new_image.at<uchar>(padding[0], i) = coefficientsLP[0] * image.at<uchar>((padding[0]<<1),i)
			+ coefficientsLP[1] * (image.at<uchar>((padding[0]<<1) + 1,i) 
			+ image.at<uchar>((padding[0]<<1) - 1,i))
			+ coefficientsLP[2] * (image.at<uchar>((padding[0]<<1) + 2,i) 
			+ image.at<uchar>((padding[0]<<1) - 2,i))
			+ coefficientsLP[3] * (image.at<uchar>((padding[0]<<1) + 3,i) 
			+ image.at<uchar>((padding[0]<<1) - 3,i))
			+ coefficientsLP[4] * (ZERO_VALUE + image.at<uchar>((padding[0]<<1) - 4,i));

		new_image.at<uchar>(padding[1], i) = coefficientsLP[0] * image.at<uchar>((padding[1]<<1),i)
			+ coefficientsLP[1] * (image.at<uchar>((padding[1]<<1) + 1,i) 
			+ image.at<uchar>((padding[1]<<1) - 1,i))
			+ coefficientsLP[2] * (ZERO_VALUE + image.at<uchar>((padding[1]<<1) - 2,i))
			+ coefficientsLP[3] * (ZERO_VALUE + image.at<uchar>((padding[1]<<1) - 3,i))
			+ coefficientsLP[4] * (ZERO_VALUE + image.at<uchar>((padding[1]<<1) - 4,i));

		new_image.at<uchar>(padding[2], i) = ZERO_VALUE
			+ coefficientsLP[1] * (ZERO_VALUE + image.at<uchar>((padding[2]<<1) - 1,i))
			+ coefficientsLP[2] * (ZERO_VALUE + image.at<uchar>((padding[2]<<1) - 2,i))
			+ coefficientsLP[3] * (ZERO_VALUE + image.at<uchar>((padding[2]<<1) - 3,i))
			+ coefficientsLP[4] * (ZERO_VALUE + image.at<uchar>((padding[2]<<1) - 4,i));
	}

	for (i = 0; i < imageWidth ; i++) {

		//Calcolo matrice di dettaglio da e dd
		new_image.at<uchar>(0+(w/2 - 1),i) = coefficientsHP[0] * new_image.at<uchar>(0,i)
			+ coefficientsHP[1] * (image.at<uchar>(1,i) + ZERO_VALUE)
			+ coefficientsHP[2] * (image.at<uchar>(2,i) + ZERO_VALUE)
			+ coefficientsHP[3] * (image.at<uchar>(3,i) + ZERO_VALUE)
			+ coefficientsHP[4] * (image.at<uchar>(4,i) + ZERO_VALUE);

		new_image.at<uchar>(1+(w/2 - 1),i) = coefficientsHP[0] * image.at<uchar>(2,i)
			+ coefficientsHP[1] * (image.at<uchar>(3,i) + image.at<uchar>(1,i))
			+ coefficientsHP[2] * (image.at<uchar>(4,i) + image.at<uchar>(0,i))
			+ coefficientsHP[3] * (image.at<uchar>(5,i) + ZERO_VALUE)
			+ coefficientsHP[4] * (image.at<uchar>(6,i) + ZERO_VALUE);

		new_image.at<uchar>(2+(w/2 - 1),i) = coefficientsHP[0] * image.at<uchar>(4,i)
			+ coefficientsHP[1] * (image.at<uchar>(5,i) + image.at<uchar>(3,i))
			+ coefficientsHP[2] * (image.at<uchar>(6,i) + image.at<uchar>(2,i))
			+ coefficientsHP[3] * (image.at<uchar>(7,i) + image.at<uchar>(1,i))
			+ coefficientsHP[4] * (image.at<uchar>(8,i) + image.at<uchar>(0,i));

		
		for (j = 3; j < (w/2) - 2; j++) {
			new_image.at<uchar>(j+(w/2 - 1),i) = coefficientsHP[0] * image.at<uchar>((j<<1),i)
				+ coefficientsHP[1] * (image.at<uchar>((j<<1) + 1,i)+image.at<uchar>((j<<1) - 1,i))
				+ coefficientsHP[2] * (image.at<uchar>((j<<1) + 2,i)+image.at<uchar>((j<<1) - 2,i))
				+ coefficientsHP[3] * (image.at<uchar>((j<<1) + 3,i)+image.at<uchar>((j<<1) - 3,i))
				+ coefficientsHP[4] * (image.at<uchar>((j<<1) + 4,i)+image.at<uchar>((j<<1) - 4,i));
		}
		
		new_image.at<uchar>(padding[0]+(w/2 - 1),i) = coefficientsHP[0] * image.at<uchar>((padding[0]<<1),i)
			+ coefficientsHP[1] * (image.at<uchar>((padding[0]<<1)+1,i) + image.at<uchar>((padding[0]<<1)-1,i))
			+ coefficientsHP[2] * (image.at<uchar>((padding[0]<<1)+2,i) + image.at<uchar>((padding[0]<<1)-2,i))
			+ coefficientsHP[3] * (image.at<uchar>((padding[0]<<1)+3,i) + image.at<uchar>((padding[0]<<1)-3,i))
			+ coefficientsHP[4] * (ZERO_VALUE + image.at<uchar>((padding[0]<<1)-4,i));
		
		new_image.at<uchar>(padding[1]+(w/2 - 1),i) = coefficientsHP[0] * image.at<uchar>((padding[1]<<1),i)
			+ coefficientsHP[1] * (image.at<uchar>((padding[1]<<1)+1,i) + image.at<uchar>((padding[1]<<1)-1,i))
			+ coefficientsHP[2] * (image.at<uchar>((padding[1]<<1)+2,i) + image.at<uchar>((padding[1]<<1)-2,i))
			+ coefficientsHP[3] * (ZERO_VALUE + image.at<uchar>((padding[1]<<1)-3,i))
			+ coefficientsHP[4] * (ZERO_VALUE + image.at<uchar>((padding[1]<<1)-4,i));


		new_image.at<uchar>(padding[2]+(w/2 - 1),i) = ZERO_VALUE
			+ coefficientsHP[1] * (ZERO_VALUE + image.at<uchar>((padding[2]<<1)-1,i))
			+ coefficientsHP[2] * (ZERO_VALUE + image.at<uchar>((padding[2]<<1)-2,i))
			+ coefficientsHP[3] * (ZERO_VALUE + image.at<uchar>((padding[2]<<1)-3,i))
			+ coefficientsHP[4] * (ZERO_VALUE + image.at<uchar>((padding[2]<<1)-4,i));
		
	}
	time_DD = omp_get_wtime();
	time_DD -= time_AA;
	final_time = omp_get_wtime();
	final_time -= initial_time;

	printf("time %lf\n", final_time);
	printf("time A e D %lf\n", time_D);
	printf("time AA e DD %lf\n", time_DD);


	//namedWindow("CDF_seriale", WINDOW_AUTOSIZE);
    //	cv::resize(new_image,new_image, cv::Size(1920,1080),0,0,cv::INTER_LINEAR);
    //	Mat output_image = cv::Mat(imageHeight,imageWidth,CV_8UC1);
    	
    //	new_image.convertTo(new_image,CV_32FC1,1/255.0);
    	
    //	cv::normalize(new_image,output_image,0,255,NORM_MINMAX,CV_8UC1);
	imshow("CDF seriale",new_image);

//	vector<int> compression_params;

//	compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);

//	compression_params.push_back(60);

//	imwrite("./immagini_modificate/CDF_9_7_seriale/CDF_9_7_seriale.jpg",new_image,compression_params);
	waitKey();

	return 0;
}




