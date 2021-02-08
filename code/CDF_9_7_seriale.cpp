#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <math.h>
#include <omp.h>

using namespace cv;
using namespace std;

#define ZERO_VALUE (double)0

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
	//Mat new_image = cv::Mat(imageHeight,imageWidth,CV_16C1);
	
	double	new_image[imageWidth][imageHeight];

	int w = imageWidth;

	double coefficientsLP[5] = { 0.602949018236,0.266864118443,-0.078223266529,-0.016864118443,0.26748757411 };
	double coefficientsHP[5] = { 1.11508705, -0.59127176314, -0.057543526229, 0.091271763114, 0.0 };
	int padding[3] = { (w / 2) - 2 ,(w / 2) - 1 ,(w / 2) };
	double initial_time = 0, final_time = 0;
	int i=0,j=0;

	double time_A=0,time_D=0,time_AA=0,time_DD=0;
	initial_time = omp_get_wtime();

	
	time_A = omp_get_wtime();
	
	//Calculations for matrix A
	for (i = 0; i < imageHeight; i++) {
	
		new_image[i][0] = coefficientsLP[0] * image.at<double>(i,0) 
			+ coefficientsLP[1] * (image.at<double>(i, 1) + ZERO_VALUE)
			+ coefficientsLP[2] * (image.at<double>(i, 2) + ZERO_VALUE)
			+ coefficientsLP[3] * (image.at<double>(i, 3) + ZERO_VALUE)
			+ coefficientsLP[4] * (image.at<double>(i, 4) + ZERO_VALUE);

		new_image[i][1] = coefficientsLP[0] * image.at<double>(i, 2) 
			+ coefficientsLP[1] * (image.at<double>(i, 3) + image.at<double>(i, 1))
			+ coefficientsLP[2] * (image.at<double>(i, 4) + image.at<double>(i, 0))
			+ coefficientsLP[3] * (image.at<double>(i, 5) + ZERO_VALUE)
			+ coefficientsLP[4] * (image.at<double>(i, 6) + ZERO_VALUE);

		new_image[i][2] = coefficientsLP[0] * image.at<double>(i, 4) 
			+ coefficientsLP[1] * (image.at<double>(i, 5) + image.at<double>(i, 3))
			+ coefficientsLP[2] * (image.at<double>(i, 6) + image.at<double>(i, 2))
			+ coefficientsLP[3] * (image.at<double>(i, 7) + image.at<double>(i, 1))
			+ coefficientsLP[4] * (image.at<double>(i, 8) + image.at<double>(i, 0));


		for (j = 3; j < (w/2) - 2; j++) {
			new_image[i][j] = coefficientsLP[0] * image.at<double>(i, (j<<1))
				+ coefficientsLP[1] * (image.at<double>(i, (j<<1) + 1) + image.at<double>(i, (j<<1) - 1))
				+ coefficientsLP[2] * (image.at<double>(i, (j<<1) + 2) + image.at<double>(i, (j<<1) - 2))
				+ coefficientsLP[3] * (image.at<double>(i, (j<<1) + 3) + image.at<double>(i, (j<<1) - 3))
				+ coefficientsLP[4] * (image.at<double>(i, (j<<1) + 4) + image.at<double>(i, (j<<1) - 4));
		}

		new_image[i][padding[0]] = coefficientsLP[0] * image.at<double>(i,(padding[0]<<1))
			+ coefficientsLP[1] * (image.at<double>(i, (padding[0]<<1) + 1) 
			+ image.at<double>(i, (padding[0]<<1) - 1))
			+ coefficientsLP[2] * (image.at<double>(i, (padding[0]<<1) + 2) 
			+ image.at<double>(i, (padding[0]<<1) - 2))
			+ coefficientsLP[3] * (image.at<double>(i, (padding[0]<<1) + 3) 
			+ image.at<double>(i, (padding[0]<<1) - 3))
			+ coefficientsLP[4] * (ZERO_VALUE + image.at<double>(i, (padding[0]<<1) - 4));

		new_image[i][padding[1]] = coefficientsLP[0] * image.at<double>(i, (padding[1]<<1))
			+ coefficientsLP[1] * (image.at<double>(i,(padding[1]<<1) + 1) 
			+ image.at<double>(i, (padding[1]<<1) - 1))
			+ coefficientsLP[2] * (ZERO_VALUE + image.at<double>(i, (padding[1]<<1) - 2))
			+ coefficientsLP[3] * (ZERO_VALUE + image.at<double>(i, (padding[1]<<1) - 3))
			+ coefficientsLP[4] * (ZERO_VALUE + image.at<double>(i, (padding[1]<<1) - 4));

		new_image[i][padding[2]] = ZERO_VALUE
			+ coefficientsLP[1] * (ZERO_VALUE + image.at<double>(i, (padding[2] << 1) - 1))
			+ coefficientsLP[2] * (ZERO_VALUE + image.at<double>(i, (padding[2] << 1) - 2))
			+ coefficientsLP[3] * (ZERO_VALUE + image.at<double>(i, (padding[2] << 1) - 3))
			+ coefficientsLP[4] * (ZERO_VALUE + image.at<double>(i, (padding[2] << 1) - 4));
	}

	//Calculations for matrix D
	for (i = 0; i < image.rows ; i++) {
	
		new_image[i][0+(w/2)] = coefficientsHP[0] * image.at<double>(i, 0)
			+ coefficientsHP[1] * (image.at<double>(i, 1) + ZERO_VALUE)
			+ coefficientsHP[2] * (image.at<double>(i, 2) + ZERO_VALUE)
			+ coefficientsHP[3] * (image.at<double>(i, 3) + ZERO_VALUE)
			+ coefficientsHP[4] * (image.at<double>(i, 4) + ZERO_VALUE);

		new_image[i][1+(w/2)] = coefficientsHP[0] * image.at<double>(i, 2)
			+ coefficientsHP[1] * (image.at<double>(i, 3) + image.at<double>(i, 1))
			+ coefficientsHP[2] * (image.at<double>(i, 4) + image.at<double>(i, 0))
			+ coefficientsHP[3] * (image.at<double>(i, 5) + ZERO_VALUE)
			+ coefficientsHP[4] * (image.at<double>(i, 6) + ZERO_VALUE);

		new_image[i][2+(w/2)] = coefficientsHP[0] * image.at<double>(i, 4)
			+ coefficientsHP[1] * (image.at<double>(i, 5) + image.at<double>(i, 3))
			+ coefficientsHP[2] * (image.at<double>(i, 6) + image.at<double>(i, 2))
			+ coefficientsHP[3] * (image.at<double>(i, 7) + image.at<double>(i, 1))
			+ coefficientsHP[4] * (image.at<double>(i, 8) + image.at<double>(i, 0));

		
		for (j = 3; j < (imageWidth/2) - 2; j++) {
			 new_image[i][j+(w/2)] = coefficientsHP[0] * image.at<double>(i, (j<<1))
				+ coefficientsHP[1] * (image.at<double>(i, (j<<1) + 1)+image.at<double>(i, (j<<1) - 1))
				+ coefficientsHP[2] * (image.at<double>(i, (j<<1) + 2)+image.at<double>(i, (j<<1) - 2))
				+ coefficientsHP[3] * (image.at<double>(i, (j<<1) + 3)+image.at<double>(i, (j<<1) - 3))
				+ coefficientsHP[4] * (image.at<double>(i, (j<<1) + 4)+image.at<double>(i, (j<<1) - 4));
		}
		
		new_image[i][padding[0]+(w/2)] = coefficientsHP[0] * image.at<double>(i, (padding[0] << 1))
			+ coefficientsHP[1] * (image.at<double>(i, (padding[0] << 1) + 1) 
				+ image.at<double>(i, (padding[0] << 1) - 1))
			+ coefficientsHP[2] * (image.at<double>(i, (padding[0] << 1) + 2) 
				+ image.at<double>(i, (padding[0] << 1) - 2))
			+ coefficientsHP[3] * (image.at<double>(i, (padding[0] << 1) + 3) 
				+ image.at<double>(i, (padding[0] << 1) - 3))
			+ coefficientsHP[4] * (ZERO_VALUE + image.at<double>(i, (padding[0] << 1) - 4));
		
		new_image[i][padding[1]+(w/2)] = coefficientsHP[0] * image.at<double>(i, (padding[1] << 1))
			+ coefficientsHP[1] * (image.at<double>(i, (padding[1]<<1)+1) + image.at<double>(i, (padding[1] << 1) - 1))
			+ coefficientsHP[2] * (ZERO_VALUE + image.at<double>(i, (padding[1] << 1) - 2))
			+ coefficientsHP[3] * (ZERO_VALUE + image.at<double>(i, (padding[1] << 1) - 3))
			+ coefficientsHP[4] * (ZERO_VALUE + image.at<double>(i, (padding[1] << 1) - 4));
		
		new_image[i][padding[2]+(w/2)] = ZERO_VALUE 
			+ coefficientsHP[1] * (ZERO_VALUE + image.at<double>(i, (padding[2]<<1) - 1))
			+ coefficientsHP[2] * (ZERO_VALUE + image.at<double>(i, (padding[2]<<1) - 2))
			+ coefficientsHP[3] * (ZERO_VALUE + image.at<double>(i, (padding[2]<<1) - 3))
			+ coefficientsHP[4] * (ZERO_VALUE + image.at<double>(i, (padding[2]<<1) - 4));
	}

//	time_D = omp_get_wtime();
//	time_D -= time_A;


//	
//	//image = new_image.clone();

//	w = imageHeight;
//	padding[0] = (w/2)-2;
//	padding[1] = (w/2)-1;
//	padding[2] = (w/2);
//	
//	time_AA =  omp_get_wtime();
	
	//Calculations for matrices AA and DA
//	for (i = 0; i < imageWidth; i++) {
//		
//		new_image.at<double>(0, i) = coefficientsLP[0] * image.at<double>(0,i) 
//			+ coefficientsLP[1] * (image.at<double>(1,i) + ZERO_VALUE)
//			+ coefficientsLP[2] * (image.at<double>(2,i) + ZERO_VALUE)
//			+ coefficientsLP[3] * (image.at<double>(3,i) + ZERO_VALUE)
//			+ coefficientsLP[4] * (image.at<double>(4,i) + ZERO_VALUE);

//		new_image.at<double>(1, i) = coefficientsLP[0] * image.at<double>(2,i) 
//			+ coefficientsLP[1] * (image.at<double>(3,i) + image.at<double>(1,i))
//			+ coefficientsLP[2] * (image.at<double>(4,i) + image.at<double>(0,i))
//			+ coefficientsLP[3] * (image.at<double>(5,i) + ZERO_VALUE)
//			+ coefficientsLP[4] * (image.at<double>(6,i) + ZERO_VALUE);

//		new_image.at<double>(2, i) = coefficientsLP[0] * image.at<double>(4,i) 
//			+ coefficientsLP[1] * (image.at<double>(5,i) + image.at<double>(3,i))
//			+ coefficientsLP[2] * (image.at<double>(6,i) + image.at<double>(2,i))
//			+ coefficientsLP[3] * (image.at<double>(7,i) + image.at<double>(1,i))
//			+ coefficientsLP[4] * (image.at<double>(8,i) + image.at<double>(0,i));


//		for (j = 3; j < (w/2) - 2; j++) {
//			new_image.at<double>(j, i) = coefficientsLP[0] * image.at<double>((j<<1),i)
//				+ coefficientsLP[1] * (image.at<double>((j<<1) + 1,i) + image.at<double>((j<<1) - 1,i))
//				+ coefficientsLP[2] * (image.at<double>((j<<1) + 2,i) + image.at<double>((j<<1) - 2,i))
//				+ coefficientsLP[3] * (image.at<double>((j<<1) + 3,i) + image.at<double>((j<<1) - 3,i))
//				+ coefficientsLP[4] * (image.at<double>((j<<1) + 4,i) + image.at<double>((j<<1) - 4,i));
//		}

//		new_image.at<double>(padding[0], i) = coefficientsLP[0] * image.at<double>((padding[0]<<1),i)
//			+ coefficientsLP[1] * (image.at<double>((padding[0]<<1) + 1,i) 
//			+ image.at<double>((padding[0]<<1) - 1,i))
//			+ coefficientsLP[2] * (image.at<double>((padding[0]<<1) + 2,i) 
//			+ image.at<double>((padding[0]<<1) - 2,i))
//			+ coefficientsLP[3] * (image.at<double>((padding[0]<<1) + 3,i) 
//			+ image.at<double>((padding[0]<<1) - 3,i))
//			+ coefficientsLP[4] * (ZERO_VALUE + image.at<double>((padding[0]<<1) - 4,i));

//		new_image.at<double>(padding[1], i) = coefficientsLP[0] * image.at<double>((padding[1]<<1),i)
//			+ coefficientsLP[1] * (image.at<double>((padding[1]<<1) + 1,i) 
//			+ image.at<double>((padding[1]<<1) - 1,i))
//			+ coefficientsLP[2] * (ZERO_VALUE + image.at<double>((padding[1]<<1) - 2,i))
//			+ coefficientsLP[3] * (ZERO_VALUE + image.at<double>((padding[1]<<1) - 3,i))
//			+ coefficientsLP[4] * (ZERO_VALUE + image.at<double>((padding[1]<<1) - 4,i));

//		new_image.at<double>(padding[2], i) = ZERO_VALUE
//			+ coefficientsLP[1] * (ZERO_VALUE + image.at<double>((padding[2]<<1) - 1,i))
//			+ coefficientsLP[2] * (ZERO_VALUE + image.at<double>((padding[2]<<1) - 2,i))
//			+ coefficientsLP[3] * (ZERO_VALUE + image.at<double>((padding[2]<<1) - 3,i))
//			+ coefficientsLP[4] * (ZERO_VALUE + image.at<double>((padding[2]<<1) - 4,i));
//	}

//	//Calculations for matrices DA and DD
//	for (i = 0; i < imageWidth ; i++) {

//		new_image.at<double>(0+(w/2 - 1),i) = coefficientsHP[0] * new_image.at<double>(0,i)
//			+ coefficientsHP[1] * (image.at<double>(1,i) + ZERO_VALUE)
//			+ coefficientsHP[2] * (image.at<double>(2,i) + ZERO_VALUE)
//			+ coefficientsHP[3] * (image.at<double>(3,i) + ZERO_VALUE)
//			+ coefficientsHP[4] * (image.at<double>(4,i) + ZERO_VALUE);

//		new_image.at<double>(1+(w/2 - 1),i) = coefficientsHP[0] * image.at<double>(2,i)
//			+ coefficientsHP[1] * (image.at<double>(3,i) + image.at<double>(1,i))
//			+ coefficientsHP[2] * (image.at<double>(4,i) + image.at<double>(0,i))
//			+ coefficientsHP[3] * (image.at<double>(5,i) + ZERO_VALUE)
//			+ coefficientsHP[4] * (image.at<double>(6,i) + ZERO_VALUE);

//		new_image.at<double>(2+(w/2 - 1),i) = coefficientsHP[0] * image.at<double>(4,i)
//			+ coefficientsHP[1] * (image.at<double>(5,i) + image.at<double>(3,i))
//			+ coefficientsHP[2] * (image.at<double>(6,i) + image.at<double>(2,i))
//			+ coefficientsHP[3] * (image.at<double>(7,i) + image.at<double>(1,i))
//			+ coefficientsHP[4] * (image.at<double>(8,i) + image.at<double>(0,i));

//		
//		for (j = 3; j < (w/2) - 2; j++) {
//			new_image.at<double>(j+(w/2 - 1),i) = coefficientsHP[0] * image.at<double>((j<<1),i)
//				+ coefficientsHP[1] * (image.at<double>((j<<1) + 1,i)+image.at<double>((j<<1) - 1,i))
//				+ coefficientsHP[2] * (image.at<double>((j<<1) + 2,i)+image.at<double>((j<<1) - 2,i))
//				+ coefficientsHP[3] * (image.at<double>((j<<1) + 3,i)+image.at<double>((j<<1) - 3,i))
//				+ coefficientsHP[4] * (image.at<double>((j<<1) + 4,i)+image.at<double>((j<<1) - 4,i));
//		}
//		
//		new_image.at<double>(padding[0]+(w/2 - 1),i) = coefficientsHP[0] * image.at<double>((padding[0]<<1),i)
//			+ coefficientsHP[1] * (image.at<double>((padding[0]<<1)+1,i) + image.at<double>((padding[0]<<1)-1,i))
//			+ coefficientsHP[2] * (image.at<double>((padding[0]<<1)+2,i) + image.at<double>((padding[0]<<1)-2,i))
//			+ coefficientsHP[3] * (image.at<double>((padding[0]<<1)+3,i) + image.at<double>((padding[0]<<1)-3,i))
//			+ coefficientsHP[4] * (ZERO_VALUE + image.at<double>((padding[0]<<1)-4,i));
//		
//		new_image.at<double>(padding[1]+(w/2 - 1),i) = coefficientsHP[0] * image.at<double>((padding[1]<<1),i)
//			+ coefficientsHP[1] * (image.at<double>((padding[1]<<1)+1,i) + image.at<double>((padding[1]<<1)-1,i))
//			+ coefficientsHP[2] * (image.at<double>((padding[1]<<1)+2,i) + image.at<double>((padding[1]<<1)-2,i))
//			+ coefficientsHP[3] * (ZERO_VALUE + image.at<double>((padding[1]<<1)-3,i))
//			+ coefficientsHP[4] * (ZERO_VALUE + image.at<double>((padding[1]<<1)-4,i));


//		new_image.at<double>(padding[2]+(w/2 - 1),i) = ZERO_VALUE
//			+ coefficientsHP[1] * (ZERO_VALUE + image.at<double>((padding[2]<<1)-1,i))
//			+ coefficientsHP[2] * (ZERO_VALUE + image.at<double>((padding[2]<<1)-2,i))
//			+ coefficientsHP[3] * (ZERO_VALUE + image.at<double>((padding[2]<<1)-3,i))
//			+ coefficientsHP[4] * (ZERO_VALUE + image.at<double>((padding[2]<<1)-4,i));
//		
//	}
	time_DD = omp_get_wtime();
	time_DD -= time_AA;
	final_time = omp_get_wtime();
	final_time -= initial_time;
	
	
	cv::Mat final_image(imageWidth,imageHeight,CV_8UC1);	
	
	
	for(int i=0;i<imageWidth;i++){
		for(int j=0;j<imageHeight;j++){
			
			final_image.at<uchar>(i,j) = (uchar)new_image[i][j];
			printf("%lf\n", new_image[i][j]);
		
		}
	}
	
	//std::memcpy(final_image.data,new_image,imageWidth*imageHeight*sizeof(double));

	printf("time %lf\n", final_time);
	printf("time A e D %lf\n", time_D);
	printf("time AA e DD %lf\n", time_DD);
	
	imshow("Serial CDDF 9/7",final_image);
	waitKey(0);

//Functions to save image on disk
//	vector<int> compression_params;

//	compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);

//	compression_params.push_back(60);

//	imwrite("./immagini_modificate/CDF_9_7_seriale/CDF_9_7_seriale.jpg",new_image,compression_params);

	return 0;
}




