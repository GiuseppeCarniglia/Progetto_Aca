#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <math.h>
#include <omp.h>

using namespace cv;
using namespace std;

#define ZERO_VALUE (double)0.0

/*
   
   
   ________________________________________________9/7-CDF Transform_________________________________________________________
   * 
   * 
  
 */


int main(int argc, char *argv[])	
{
	const char* imgName = argv[1];
//	double x [][10] = {{1,2,3,4,5,6,7,8,9,10},
//			 {11,12,13,15,14,16,17,18,19,20}};
	//Mat image = cv::Mat(2,10,CV_64FC1,x);
	
	Mat image = imread(imgName, IMREAD_GRAYSCALE);
	
	
	if(image.empty()){
		perror("Empty image\n");
		return 0;
	}
	
	image.convertTo(image,CV_64FC1);
	
	int imageWidth = image.size().width;
	int imageHeight = image.size().height;
//	//Mat new_image = cv::Mat(imageHeight,imageWidth,CV_16C1);
//	

	printf("%d %d\n",imageHeight,imageWidth);
	double	*row,**new_image;
	
	row = (double*)malloc(imageHeight * imageWidth * sizeof(*row));
	
	new_image = (double**)malloc(imageHeight * sizeof(*new_image));	
	
	for(int i=0;i<imageHeight;i++, row += imageWidth){
		new_image[i] = row;
	}
	
	Mat tmp_image = cv::Mat(imageHeight,imageWidth,CV_64FC1);

	int w = imageWidth;

	double coefficientsLP[5] = { 0.602949018236,0.266864118443,-0.078223266529,-0.016864118443,0.26748757411 };
	double coefficientsHP[5] = { 1.11508705, -0.59127176314, -0.057543526229, 0.091271763114, 0.0 };
	
		double coefficientsLPR[5] = { 1.11508705, 0.59127176314, 0.057543526229, -0.091271763114, 0.0};
	double coefficientsHPR[5] = { 0.602949018236, -0.266864118443,-0.078223266529,	0.016864118443,0.26748757411 };
	
	int padding[3] = { (w / 2) - 2 ,(w / 2) - 1 ,(w / 2) };
	double initial_time = 0, final_time = 0;
	int i=0,j=0;

	double time_A=0,time_D=0,time_AA=0,time_DD=0;
	initial_time = omp_get_wtime();

	
	time_A = omp_get_wtime();
	
	
	for(int i=0;i<imageHeight;i++){
		for(int j=0;j<imageWidth;j++){
			
			new_image[i][j] = 0.0;
		}
	}
	
//	Calculations for matrix A
	for (i = 0; i < imageHeight; i++) {
	
		new_image[i][0] = coefficientsLP[0] * image.at<double>(i,0) 
			+ coefficientsLP[1] * (image.at<double>(i, 1) + ZERO_VALUE)
			+ coefficientsLP[2] * (image.at<double>(i, 2) + ZERO_VALUE)
			+ coefficientsLP[3] * (image.at<double>(i, 3) + ZERO_VALUE)
			+ coefficientsLP[4] * (image.at<double>(i, 4) + ZERO_VALUE);
		//printf("i: %d new_image: %lf\n",i,new_image[i][0]);

		new_image[i][1] = coefficientsLP[0] * image.at<double>(i, 2) 
			+ coefficientsLP[1] * (image.at<double>(i, 3) + image.at<double>(i, 1))
			+ coefficientsLP[2] * (image.at<double>(i, 4) + image.at<double>(i, 0))
			+ coefficientsLP[3] * (image.at<double>(i, 5) + ZERO_VALUE)
			+ coefficientsLP[4] * (image.at<double>(i, 6) + ZERO_VALUE);

		//printf("i: %d new_image: %lf\n",i,new_image[i][0]);
		
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
//			if(new_image[i][j]!=0){	
//				printf("i: %d j: %d new_image: %lf\n",i,j,new_image[i][j]);
//				getchar();		
//			}
			
			
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
	for (i = 0; i < imageHeight ; i++) {
	
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

		
		for (j = 3; j < (w/2) - 2; j++) {
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

	for(int i=0;i<imageHeight;i++){
		for(int j=0;j<imageWidth;j++){
			 tmp_image.at<double>(i,j) = new_image[i][j];
		}
	}

	w = imageHeight;
	padding[0] = (w/2)-2;
	padding[1] = (w/2)-1;
	padding[2] = (w/2);
	
	time_AA =  omp_get_wtime();
	
	//Calculations for matrices AA and DA
	for (i = 0; i < imageWidth; i++) {
		
		new_image[0][i] = coefficientsLP[0] * tmp_image.at<double>(0,i) 
			+ coefficientsLP[1] * (tmp_image.at<double>(1,i) + ZERO_VALUE)
			+ coefficientsLP[2] * (tmp_image.at<double>(2,i) + ZERO_VALUE)
			+ coefficientsLP[3] * (tmp_image.at<double>(3,i) + ZERO_VALUE)
			+ coefficientsLP[4] * (tmp_image.at<double>(4,i) + ZERO_VALUE);

		new_image[1][i] = coefficientsLP[0] * tmp_image.at<double>(2,i) 
			+ coefficientsLP[1] * (tmp_image.at<double>(3,i) + tmp_image.at<double>(1,i))
			+ coefficientsLP[2] * (tmp_image.at<double>(4,i) + tmp_image.at<double>(0,i))
			+ coefficientsLP[3] * (tmp_image.at<double>(5,i) + ZERO_VALUE)
			+ coefficientsLP[4] * (tmp_image.at<double>(6,i) + ZERO_VALUE);

		new_image[2][i] = coefficientsLP[0] * tmp_image.at<double>(4,i) 
			+ coefficientsLP[1] * (tmp_image.at<double>(5,i) + tmp_image.at<double>(3,i))
			+ coefficientsLP[2] * (tmp_image.at<double>(6,i) + tmp_image.at<double>(2,i))
			+ coefficientsLP[3] * (tmp_image.at<double>(7,i) + tmp_image.at<double>(1,i))
			+ coefficientsLP[4] * (tmp_image.at<double>(8,i) + tmp_image.at<double>(0,i));


		for (j = 3; j < (w/2) - 2; j++) {
			new_image[j][i] = coefficientsLP[0] * tmp_image.at<double>((j<<1),i)
				+ coefficientsLP[1] * (tmp_image.at<double>((j<<1) + 1,i) + tmp_image.at<double>((j<<1) - 1,i))
				+ coefficientsLP[2] * (tmp_image.at<double>((j<<1) + 2,i) + tmp_image.at<double>((j<<1) - 2,i))
				+ coefficientsLP[3] * (tmp_image.at<double>((j<<1) + 3,i) + tmp_image.at<double>((j<<1) - 3,i))
				+ coefficientsLP[4] * (tmp_image.at<double>((j<<1) + 4,i) + tmp_image.at<double>((j<<1) - 4,i));
		}

		new_image[padding[0]][i] = coefficientsLP[0] * tmp_image.at<double>((padding[0]<<1),i)
			+ coefficientsLP[1] * (tmp_image.at<double>((padding[0]<<1) + 1,i) 
			+ tmp_image.at<double>((padding[0]<<1) - 1,i))
			+ coefficientsLP[2] * (tmp_image.at<double>((padding[0]<<1) + 2,i) 
			+ tmp_image.at<double>((padding[0]<<1) - 2,i))
			+ coefficientsLP[3] * (tmp_image.at<double>((padding[0]<<1) + 3,i) 
			+ tmp_image.at<double>((padding[0]<<1) - 3,i))
			+ coefficientsLP[4] * (ZERO_VALUE + tmp_image.at<double>((padding[0]<<1) - 4,i));

		new_image[padding[1]][i] = coefficientsLP[0] * tmp_image.at<double>((padding[1]<<1),i)
			+ coefficientsLP[1] * (tmp_image.at<double>((padding[1]<<1) + 1,i) 
			+ tmp_image.at<double>((padding[1]<<1) - 1,i))
			+ coefficientsLP[2] * (ZERO_VALUE + tmp_image.at<double>((padding[1]<<1) - 2,i))
			+ coefficientsLP[3] * (ZERO_VALUE + tmp_image.at<double>((padding[1]<<1) - 3,i))
			+ coefficientsLP[4] * (ZERO_VALUE + tmp_image.at<double>((padding[1]<<1) - 4,i));

		new_image[padding[2]][i] = ZERO_VALUE
			+ coefficientsLP[1] * (ZERO_VALUE + tmp_image.at<double>((padding[2]<<1) - 1,i))
			+ coefficientsLP[2] * (ZERO_VALUE + tmp_image.at<double>((padding[2]<<1) - 2,i))
			+ coefficientsLP[3] * (ZERO_VALUE + tmp_image.at<double>((padding[2]<<1) - 3,i))
			+ coefficientsLP[4] * (ZERO_VALUE + tmp_image.at<double>((padding[2]<<1) - 4,i));
	}

//	//Calculations for matrices DA and DD
	for (i = 0; i < imageWidth ; i++) {

		new_image[0+(w/2 - 1)][i] = coefficientsHP[0] * tmp_image.at<double>(0,i)
			+ coefficientsHP[1] * (tmp_image.at<double>(1,i) + ZERO_VALUE)
			+ coefficientsHP[2] * (tmp_image.at<double>(2,i) + ZERO_VALUE)
			+ coefficientsHP[3] * (tmp_image.at<double>(3,i) + ZERO_VALUE)
			+ coefficientsHP[4] * (tmp_image.at<double>(4,i) + ZERO_VALUE);

		new_image[1+(w/2 - 1)][i] = coefficientsHP[0] * tmp_image.at<double>(2,i)
			+ coefficientsHP[1] * (tmp_image.at<double>(3,i) + tmp_image.at<double>(1,i))
			+ coefficientsHP[2] * (tmp_image.at<double>(4,i) + tmp_image.at<double>(0,i))
			+ coefficientsHP[3] * (tmp_image.at<double>(5,i) + ZERO_VALUE)
			+ coefficientsHP[4] * (tmp_image.at<double>(6,i) + ZERO_VALUE);

		new_image[2+(w/2 - 1)][i] = coefficientsHP[0] * tmp_image.at<double>(4,i)
			+ coefficientsHP[1] * (tmp_image.at<double>(5,i) + tmp_image.at<double>(3,i))
			+ coefficientsHP[2] * (tmp_image.at<double>(6,i) + tmp_image.at<double>(2,i))
			+ coefficientsHP[3] * (tmp_image.at<double>(7,i) + tmp_image.at<double>(1,i))
			+ coefficientsHP[4] * (tmp_image.at<double>(8,i) + tmp_image.at<double>(0,i));

		
		for (j = 3; j < (w/2) - 2; j++) {
			new_image[j+(w/2 - 1)][i] = coefficientsHP[0] * tmp_image.at<double>((j<<1),i)
				+ coefficientsHP[1] * (tmp_image.at<double>((j<<1) + 1,i)+tmp_image.at<double>((j<<1) - 1,i))
				+ coefficientsHP[2] * (tmp_image.at<double>((j<<1) + 2,i)+tmp_image.at<double>((j<<1) - 2,i))
				+ coefficientsHP[3] * (tmp_image.at<double>((j<<1) + 3,i)+tmp_image.at<double>((j<<1) - 3,i))
				+ coefficientsHP[4] * (tmp_image.at<double>((j<<1) + 4,i)+tmp_image.at<double>((j<<1) - 4,i));
		}
		
		new_image[padding[0]+(w/2 - 1)][i] = coefficientsHP[0] * tmp_image.at<double>((padding[0]<<1),i)
			+ coefficientsHP[1] * (tmp_image.at<double>((padding[0]<<1)+1,i) + tmp_image.at<double>((padding[0]<<1)-1,i))
			+ coefficientsHP[2] * (tmp_image.at<double>((padding[0]<<1)+2,i) + tmp_image.at<double>((padding[0]<<1)-2,i))
			+ coefficientsHP[3] * (tmp_image.at<double>((padding[0]<<1)+3,i) + tmp_image.at<double>((padding[0]<<1)-3,i))
			+ coefficientsHP[4] * (ZERO_VALUE + tmp_image.at<double>((padding[0]<<1)-4,i));
		
		new_image[padding[1]+(w/2 - 1)][i] = coefficientsHP[0] * tmp_image.at<double>((padding[1]<<1),i)
			+ coefficientsHP[1] * (tmp_image.at<double>((padding[1]<<1)+1,i) + tmp_image.at<double>((padding[1]<<1)-1,i))
			+ coefficientsHP[2] * (tmp_image.at<double>((padding[1]<<1)+2,i) + tmp_image.at<double>((padding[1]<<1)-2,i))
			+ coefficientsHP[3] * (ZERO_VALUE + tmp_image.at<double>((padding[1]<<1)-3,i))
			+ coefficientsHP[4] * (ZERO_VALUE + tmp_image.at<double>((padding[1]<<1)-4,i));


		new_image[padding[2]+(w/2 - 1)][i] = ZERO_VALUE
			+ coefficientsHP[1] * (ZERO_VALUE + tmp_image.at<double>((padding[2]<<1)-1,i))
			+ coefficientsHP[2] * (ZERO_VALUE + tmp_image.at<double>((padding[2]<<1)-2,i))
			+ coefficientsHP[3] * (ZERO_VALUE + tmp_image.at<double>((padding[2]<<1)-3,i))
			+ coefficientsHP[4] * (ZERO_VALUE + tmp_image.at<double>((padding[2]<<1)-4,i));
		
	}
//	time_DD = omp_get_wtime();
//	time_DD -= time_AA;
//	final_time = omp_get_wtime();
//	final_time -= initial_time;
//	
//	

	for(int i=0;i<imageHeight;i++){
		for(int j=0;j<imageWidth;j++){
			 tmp_image.at<double>(i,j) = new_image[i][j];
		}
	}
	
	for(int i=0;i<imageHeight;i++){
		for(int j=0;j<imageWidth;j++){
			
			new_image[i][j] = 0.0;
		}
	}
	
	//Antitrasformata per immagini AA and AD
	for (i = 0; i < imageWidth; i++) {
		
		new_image[0][i] = coefficientsLPR[0] * tmp_image.at<double>(0,i) 
			+ coefficientsLPR[1] * (tmp_image.at<double>(1,i) + ZERO_VALUE)
			+ coefficientsLPR[2] * (tmp_image.at<double>(2,i) + ZERO_VALUE)
			+ coefficientsLPR[3] * (tmp_image.at<double>(3,i) + ZERO_VALUE)
			+ coefficientsLPR[4] * (tmp_image.at<double>(4,i) + ZERO_VALUE);

		new_image[2*1][i] = coefficientsLPR[0] * tmp_image.at<double>(2,i) 
			+ coefficientsLPR[1] * (tmp_image.at<double>(3,i) + tmp_image.at<double>(1,i))
			+ coefficientsLPR[2] * (tmp_image.at<double>(4,i) + tmp_image.at<double>(0,i))
			+ coefficientsLPR[3] * (tmp_image.at<double>(5,i) + ZERO_VALUE)
			+ coefficientsLPR[4] * (tmp_image.at<double>(6,i) + ZERO_VALUE);

		new_image[2*2][i] = coefficientsLPR[0] * tmp_image.at<double>(4,i) 
			+ coefficientsLPR[1] * (tmp_image.at<double>(5,i) + tmp_image.at<double>(3,i))
			+ coefficientsLPR[2] * (tmp_image.at<double>(6,i) + tmp_image.at<double>(2,i))
			+ coefficientsLPR[3] * (tmp_image.at<double>(7,i) + tmp_image.at<double>(1,i))
			+ coefficientsLPR[4] * (tmp_image.at<double>(8,i) + tmp_image.at<double>(0,i));
			
		new_image[2*3][i] = coefficientsLPR[0] * tmp_image.at<double>(6,i) 
			+ coefficientsLPR[1] * (tmp_image.at<double>(7,i) + tmp_image.at<double>(5,i))
			+ coefficientsLPR[2] * (tmp_image.at<double>(8,i) + tmp_image.at<double>(4,i))
			+ coefficientsLPR[3] * (tmp_image.at<double>(9,i) + tmp_image.at<double>(3,i))
			+ coefficientsLPR[4] * (tmp_image.at<double>(10,i) + tmp_image.at<double>(2,i));


		for (j = 4; j < (w/2) - 2; j++) {
			new_image[2*j][i] = coefficientsLPR[0] * tmp_image.at<double>((j),i)
				+ coefficientsLPR[1] * (tmp_image.at<double>((j) + 1,i) + tmp_image.at<double>((j) - 1,i))
				+ coefficientsLPR[2] * (tmp_image.at<double>((j) + 2,i) + tmp_image.at<double>((j) - 2,i))
				+ coefficientsLPR[3] * (tmp_image.at<double>((j) + 3,i) + tmp_image.at<double>((j) - 3,i))
				+ coefficientsLPR[4] * (tmp_image.at<double>((j) + 4,i) + tmp_image.at<double>((j) - 4,i));
		}

		new_image[2*padding[0]][i] = coefficientsLPR[0] * tmp_image.at<double>((padding[0]),i)
			+ coefficientsLPR[1] * (tmp_image.at<double>((padding[0]) + 1,i) 
			+ tmp_image.at<double>((padding[0]) - 1,i))
			+ coefficientsLPR[2] * (tmp_image.at<double>((padding[0]) + 2,i) 
			+ tmp_image.at<double>((padding[0]) - 2,i))
			+ coefficientsLPR[3] * (tmp_image.at<double>((padding[0]) + 3,i) 
			+ tmp_image.at<double>((padding[0]) - 3,i))
			+ coefficientsLPR[4] * (ZERO_VALUE + tmp_image.at<double>((padding[0]) - 4,i));

		new_image[2*padding[1]][i] = coefficientsLPR[0] * tmp_image.at<double>((padding[1]),i)
			+ coefficientsLPR[1] * (tmp_image.at<double>((padding[1]) + 1,i) 
			+ tmp_image.at<double>((padding[1]) - 1,i))
			+ coefficientsLPR[2] * (ZERO_VALUE + tmp_image.at<double>((padding[1]) - 2,i))
			+ coefficientsLPR[3] * (ZERO_VALUE + tmp_image.at<double>((padding[1]) - 3,i))
			+ coefficientsLPR[4] * (ZERO_VALUE + tmp_image.at<double>((padding[1]) - 4,i));

		new_image[2*padding[2]-1][i] = ZERO_VALUE
			+ coefficientsLPR[1] * (ZERO_VALUE + tmp_image.at<double>((padding[2]) - 1,i))
			+ coefficientsLPR[2] * (ZERO_VALUE + tmp_image.at<double>((padding[2]) - 2,i))
			+ coefficientsLPR[3] * (ZERO_VALUE + tmp_image.at<double>((padding[2]) - 3,i))
			+ coefficientsLPR[4] * (ZERO_VALUE + tmp_image.at<double>((padding[2]) - 4,i));
	}	

	//Antitrasformata immagini DA e DD
	for (i = 0; i < imageWidth ; i++) {

		new_image[0][i] += coefficientsHPR[0] * tmp_image.at<double>(0+(w/2 - 1),i)
			+ coefficientsHPR[1] * (tmp_image.at<double>(1+(w/2 - 1),i) + ZERO_VALUE)
			+ coefficientsHPR[2] * (tmp_image.at<double>(2+(w/2 - 1),i) + ZERO_VALUE)
			+ coefficientsHPR[3] * (tmp_image.at<double>(3+(w/2 - 1),i) + ZERO_VALUE)
			+ coefficientsHPR[4] * (tmp_image.at<double>(4+(w/2 - 1),i) + ZERO_VALUE);

		new_image[2*1][i] += coefficientsHPR[0] * tmp_image.at<double>(2+(w/2 - 1),i)
			+ coefficientsHPR[1] * (tmp_image.at<double>(3+(w/2 - 1),i) + tmp_image.at<double>(1+(w/2 - 1),i))
			+ coefficientsHPR[2] * (tmp_image.at<double>(4+(w/2 - 1),i) + tmp_image.at<double>(0+(w/2 - 1),i))
			+ coefficientsHPR[3] * (tmp_image.at<double>(5+(w/2 - 1),i) + ZERO_VALUE)
			+ coefficientsHPR[4] * (tmp_image.at<double>(6+(w/2 - 1),i) + ZERO_VALUE);

		new_image[2*2][i] += coefficientsHPR[0] * tmp_image.at<double>(4+(w/2 - 1),i)
			+ coefficientsHPR[1] * (tmp_image.at<double>(5+(w/2 - 1),i) + tmp_image.at<double>(3+(w/2 - 1),i))
			+ coefficientsHPR[2] * (tmp_image.at<double>(6+(w/2 - 1),i) + tmp_image.at<double>(2+(w/2 - 1),i))
			+ coefficientsHPR[3] * (tmp_image.at<double>(7+(w/2 - 1),i) + tmp_image.at<double>(1+(w/2 - 1),i))
			+ coefficientsHPR[4] * (tmp_image.at<double>(8+(w/2 - 1),i) + tmp_image.at<double>(0+(w/2 - 1),i));

		
		for (j = 3; j < (w/2) - 2; j++) {
			new_image[2*j][i] += coefficientsHPR[0] * tmp_image.at<double>((j+(w/2 - 1)),i)
				+ coefficientsHPR[1] * (tmp_image.at<double>((j+(w/2 - 1)) + 1,i)+tmp_image.at<double>((j+(w/2 - 1)) - 1,i))
				+ coefficientsHPR[2] * (tmp_image.at<double>((j+(w/2 - 1)) + 2,i)+tmp_image.at<double>((j+(w/2 - 1)) - 2,i))
				+ coefficientsHPR[3] * (tmp_image.at<double>((j+(w/2 - 1)) + 3,i)+tmp_image.at<double>((j+(w/2 - 1)) - 3,i))
				+ coefficientsHPR[4] * (tmp_image.at<double>((j+(w/2 - 1)) + 4,i)+tmp_image.at<double>((j+(w/2 - 1)) - 4,i));
		}
		
		new_image[2*padding[0]][i] += coefficientsHPR[0] * tmp_image.at<double>((padding[0]+(w/2 - 1)),i)
			+ coefficientsHPR[1] * (tmp_image.at<double>((padding[0]+(w/2 - 1))+1,i) + tmp_image.at<double>((padding[0]+(w/2 - 1))-1,i))
			+ coefficientsHPR[2] * (tmp_image.at<double>((padding[0]+(w/2 - 1))+2,i) + tmp_image.at<double>((padding[0]+(w/2 - 1))-2,i))
			+ coefficientsHPR[3] * (tmp_image.at<double>((padding[0]+(w/2 - 1))+3,i) + tmp_image.at<double>((padding[0]+(w/2 - 1))-3,i))
			+ coefficientsHPR[4] * (ZERO_VALUE + tmp_image.at<double>((padding[0]+(w/2 - 1))-4,i));
		
		new_image[2*padding[1]][i] += coefficientsHPR[0] * tmp_image.at<double>((padding[1]+(w/2 - 1)),i)
			+ coefficientsHPR[1] * (tmp_image.at<double>((padding[1]+(w/2 - 1))+1,i) + tmp_image.at<double>((padding[1]+(w/2 - 1))-1,i))
			+ coefficientsHPR[2] * (tmp_image.at<double>((padding[1]+(w/2 - 1))+2,i) + tmp_image.at<double>((padding[1]+(w/2 - 1))-2,i))
			+ coefficientsHPR[3] * (ZERO_VALUE + tmp_image.at<double>((padding[1]+(w/2 - 1))-3,i))
			+ coefficientsHPR[4] * (ZERO_VALUE + tmp_image.at<double>((padding[1]+(w/2 - 1))-4,i));


		new_image[2*padding[2]- 1][i] += ZERO_VALUE
			+ coefficientsHPR[1] * (ZERO_VALUE + tmp_image.at<double>((padding[2]+(w/2 - 1))-1,i))
			+ coefficientsHPR[2] * (ZERO_VALUE + tmp_image.at<double>((padding[2]+(w/2 - 1))-2,i))
			+ coefficientsHPR[3] * (ZERO_VALUE + tmp_image.at<double>((padding[2]+(w/2 - 1))-3,i))
			+ coefficientsHPR[4] * (ZERO_VALUE + tmp_image.at<double>((padding[2]+(w/2 - 1))-4,i));
		
	}
	
	for(int i=0;i<imageHeight;i++){
		for(int j=0;j<imageWidth;j++){
			 tmp_image.at<double>(i,j) = new_image[i][j];
		}
	}
	
	for(int i=0;i<imageHeight;i++){
		for(int j=0;j<imageWidth;j++){
			
			new_image[i][j] = 0.0;
		}
	}

	w = imageWidth;
	padding[0] = (w/2)-2;
	padding[1] = (w/2)-1;
	padding[2] = (w/2);

	//Antitrasformata immagine A
	for (i = 0; i < imageHeight; i++) {

	new_image[i][2*0] = coefficientsLPR[0] * tmp_image.at<double>(i,0) 
		+ coefficientsLPR[1] * (tmp_image.at<double>(i, 1) + ZERO_VALUE)
		+ coefficientsLPR[2] * (tmp_image.at<double>(i, 2) + ZERO_VALUE)
		+ coefficientsLPR[3] * (tmp_image.at<double>(i, 3) + ZERO_VALUE)
		+ coefficientsLPR[4] * (tmp_image.at<double>(i, 4) + ZERO_VALUE);
	//printf("i: %d new_image: %lf\n",i,new_image[i][0]);

	new_image[i][2*1] = coefficientsLPR[0] * tmp_image.at<double>(i, 2) 
		+ coefficientsLPR[1] * (tmp_image.at<double>(i, 3) + tmp_image.at<double>(i, 1))
		+ coefficientsLPR[2] * (tmp_image.at<double>(i, 4) + tmp_image.at<double>(i, 0))
		+ coefficientsLPR[3] * (tmp_image.at<double>(i, 5) + ZERO_VALUE)
		+ coefficientsLPR[4] * (tmp_image.at<double>(i, 6) + ZERO_VALUE);

	//printf("i: %d new_image: %lf\n",i,new_image[i][0]);
	
	new_image[i][2*2] = coefficientsLPR[0] * tmp_image.at<double>(i, 4) 
		+ coefficientsLPR[1] * (tmp_image.at<double>(i, 5) + tmp_image.at<double>(i, 3))
		+ coefficientsLPR[2] * (tmp_image.at<double>(i, 6) + tmp_image.at<double>(i, 2))
		+ coefficientsLPR[3] * (tmp_image.at<double>(i, 7) + tmp_image.at<double>(i, 1))
		+ coefficientsLPR[4] * (tmp_image.at<double>(i, 8) + tmp_image.at<double>(i, 0));


	for (j = 3; j < (w/2) - 2; j++) {
		new_image[i][2*j] = coefficientsLPR[0] * tmp_image.at<double>(i, (j))
				+ coefficientsLPR[1] * (tmp_image.at<double>(i, (j) + 1) + tmp_image.at<double>(i, (j) - 1))
				+ coefficientsLPR[2] * (tmp_image.at<double>(i, (j) + 2) + tmp_image.at<double>(i, (j) - 2))
				+ coefficientsLPR[3] * (tmp_image.at<double>(i, (j) + 3) + tmp_image.at<double>(i, (j) - 3))
				+ coefficientsLPR[4] * (tmp_image.at<double>(i, (j) + 4) + tmp_image.at<double>(i, (j) - 4));
			
			
		}

		new_image[i][2*padding[0]] = coefficientsLPR[0] * tmp_image.at<double>(i,(padding[0]<<1))
			+ coefficientsLPR[1] * (tmp_image.at<double>(i, (padding[0]) + 1) 
			+ tmp_image.at<double>(i, (padding[0]) - 1))
			+ coefficientsLPR[2] * (tmp_image.at<double>(i, (padding[0]<<1) + 2) 
			+ tmp_image.at<double>(i, (padding[0]) - 2))
			+ coefficientsLPR[3] * (tmp_image.at<double>(i, (padding[0]<<1) + 3) 
			+ tmp_image.at<double>(i, (padding[0]) - 3))
			+ coefficientsLPR[4] * (ZERO_VALUE + tmp_image.at<double>(i, (padding[0]) - 4));

		new_image[i][2*padding[1]] = coefficientsLPR[0] * tmp_image.at<double>(i, (padding[1]<<1))
			+ coefficientsLPR[1] * (tmp_image.at<double>(i,(padding[1]) + 1) 
			+ tmp_image.at<double>(i, (padding[1]) - 1))
			+ coefficientsLPR[2] * (ZERO_VALUE + tmp_image.at<double>(i, (padding[1]) - 2))
			+ coefficientsLPR[3] * (ZERO_VALUE + tmp_image.at<double>(i, (padding[1]) - 3))
			+ coefficientsLPR[4] * (ZERO_VALUE + tmp_image.at<double>(i, (padding[1]) - 4));

		new_image[i][2*padding[2]] = ZERO_VALUE
			+ coefficientsLPR[1] * (ZERO_VALUE + tmp_image.at<double>(i, (padding[2]) - 1))
			+ coefficientsLPR[2] * (ZERO_VALUE + tmp_image.at<double>(i, (padding[2]) - 2))
			+ coefficientsLPR[3] * (ZERO_VALUE + tmp_image.at<double>(i, (padding[2]) - 3))
			+ coefficientsLPR[4] * (ZERO_VALUE + tmp_image.at<double>(i, (padding[2]) - 4));
	}
	
	
	//Antitrasformata immagine D
	for (i = 0; i < imageHeight ; i++) {

	new_image[i][0] += coefficientsHPR[0] * tmp_image.at<double>(i, 0+(w/2))
		+ coefficientsHPR[1] * (tmp_image.at<double>(i, 1+(w/2)) + ZERO_VALUE)
		+ coefficientsHPR[2] * (tmp_image.at<double>(i, 2+(w/2)) + ZERO_VALUE)
		+ coefficientsHPR[3] * (tmp_image.at<double>(i, 3+(w/2)) + ZERO_VALUE)
		+ coefficientsHPR[4] * (tmp_image.at<double>(i, 4+(w/2)) + ZERO_VALUE);

	new_image[i][2*1] += coefficientsHPR[0] * tmp_image.at<double>(i, 2+(w/2))
		+ coefficientsHPR[1] * (tmp_image.at<double>(i, 3+(w/2)) + tmp_image.at<double>(i, 1+(w/2)))
		+ coefficientsHPR[2] * (tmp_image.at<double>(i, 4+(w/2)) + tmp_image.at<double>(i, 0+(w/2)))
		+ coefficientsHPR[3] * (tmp_image.at<double>(i, 5+(w/2)) + ZERO_VALUE)
		+ coefficientsHPR[4] * (tmp_image.at<double>(i, 6+(w/2)) + ZERO_VALUE);

	new_image[i][2*2] += coefficientsHPR[0] * tmp_image.at<double>(i, 4+(w/2))
		+ coefficientsHPR[1] * (tmp_image.at<double>(i, 5+(w/2)) + tmp_image.at<double>(i, 3+(w/2)))
		+ coefficientsHPR[2] * (tmp_image.at<double>(i, 6+(w/2)) + tmp_image.at<double>(i, 2+(w/2)))
		+ coefficientsHPR[3] * (tmp_image.at<double>(i, 7+(w/2)) + tmp_image.at<double>(i, 1+(w/2)))
		+ coefficientsHPR[4] * (tmp_image.at<double>(i, 8+(w/2)) + tmp_image.at<double>(i, 0+(w/2)));

	
	for (j = 3; j < (w/2) - 2; j++) {
		 new_image[i][2*j] += coefficientsHPR[0] * tmp_image.at<double>(i, (j+(w/2)))
				+ coefficientsHPR[1] * (tmp_image.at<double>(i, (j+(w/2)) + 1)+tmp_image.at<double>(i, (j+(w/2)) - 1))
				+ coefficientsHPR[2] * (tmp_image.at<double>(i, (j+(w/2)) + 2)+tmp_image.at<double>(i, (j+(w/2)) - 2))
				+ coefficientsHPR[3] * (tmp_image.at<double>(i, (j+(w/2)) + 3)+tmp_image.at<double>(i, (j+(w/2)) - 3))
				+ coefficientsHPR[4] * (tmp_image.at<double>(i, (j+(w/2)) + 4)+tmp_image.at<double>(i, (j+(w/2)) - 4));
				
		}
		
		new_image[i][2*padding[0]] += coefficientsHPR[0] * tmp_image.at<double>(i, (padding[0] << 1))
			+ coefficientsHPR[1] * (tmp_image.at<double>(i, (padding[0] << 1) + 1) 
				+ tmp_image.at<double>(i, (padding[0] << 1) - 1))
			+ coefficientsHPR[2] * (tmp_image.at<double>(i, (padding[0] << 1) + 2) 
				+ tmp_image.at<double>(i, (padding[0] << 1) - 2))
			+ coefficientsHPR[3] * (tmp_image.at<double>(i, (padding[0] << 1) + 3) 
				+ tmp_image.at<double>(i, (padding[0] << 1) - 3))
			+ coefficientsHPR[4] * (ZERO_VALUE + tmp_image.at<double>(i, (padding[0] << 1) - 4));
		
		new_image[i][2*padding[1]] += coefficientsHPR[0] * tmp_image.at<double>(i, (padding[1] << 1))
			+ coefficientsHPR[1] * (tmp_image.at<double>(i, (padding[1]<<1)+1) + tmp_image.at<double>(i, (padding[1] << 1) - 1))
			+ coefficientsHPR[2] * (ZERO_VALUE + tmp_image.at<double>(i, (padding[1] << 1) - 2))
			+ coefficientsHPR[3] * (ZERO_VALUE + tmp_image.at<double>(i, (padding[1] << 1) - 3))
			+ coefficientsHPR[4] * (ZERO_VALUE + tmp_image.at<double>(i, (padding[1] << 1) - 4));
		new_image[i][2*padding[2]] += ZERO_VALUE 
			+ coefficientsHPR[1] * (ZERO_VALUE + tmp_image.at<double>(i, (padding[2]<<1) - 1))
			+ coefficientsHPR[2] * (ZERO_VALUE + tmp_image.at<double>(i, (padding[2]<<1) - 2))
			+ coefficientsHPR[3] * (ZERO_VALUE + tmp_image.at<double>(i, (padding[2]<<1) - 3))
			+ coefficientsHPR[4] * (ZERO_VALUE + tmp_image.at<double>(i, (padding[2]<<1) - 4));
	}






	cv::Mat final_image(imageHeight,imageWidth,CV_8UC1);
//	
	for(int i=0;i<imageHeight;i++){
		for(int j=0;j<imageWidth;j++){
			
			 tmp_image.at<double>(i,j) = new_image[i][j];
		}
	}	
//	
//	
//	for(int i=0;i<imageHeight;i++){
//		for(int j=0;j<imageWidth;j++){
//			
//			final_image.at<uchar>(i,j) = (uchar)new_image[i][j];
//			printf("%lf\n",new_image[i][j]);
//		
//		}
//	}
//	
//	
	cv::normalize(tmp_image,final_image,0,255,NORM_MINMAX,CV_8UC1);
//	
//	printf("time %lf\n", final_time);
//	printf("time A e D %lf\n", time_D);
//	printf("time AA e DD %lf\n", time_DD);
//	

	
	namedWindow("Serial CDF 9/7",WINDOW_NORMAL);
	cv::resizeWindow("Serial CDF 9/7",1920,1080);
	imshow("Serial CDF 9/7",final_image);
	waitKey(0);

//Functions to save image on disk
//	vector<int> compression_params;

//	compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);

//	compression_params.push_back(60);

//	imwrite("./immagini_modificate/CDF_9_7_seriale/CDF_9_7_seriale.jpg",final_image,compression_params);

	free(*new_image);
	free(new_image);

	return 0;
}




