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


int main(int argc, char* argv[])
{
	const char* imgName = argv[1];
	Mat image = imread(imgName, IMREAD_GRAYSCALE);
	int imageWidth = image.size().width;
	int imageHeight = image.size().height;
	
	image.convertTo(image,CV_64FC1);
	

	Mat new_image = cv::Mat(imageHeight,imageWidth,CV_64FC1);

	Mat tmp_image_1 = cv::Mat(imageHeight,imageWidth,CV_64FC1);
	Mat tmp_image_2 = cv::Mat(imageHeight,imageWidth,CV_64FC1);

	double coefficientsLP[5] = { 0.602949018236,0.266864118443,-0.078223266529,-0.016864118443,0.26748757411 };
	double coefficientsHP[5] = { 1.11508705, -0.59127176314, -0.057543526229, 0.091271763114, 0.0 };

	int padding[3] = {(imageWidth/2) - 2 ,(imageWidth/2) - 1 ,(imageWidth/2)};


	double initial_time = 0, final_time = 0;
	
	if(image.empty()){
		perror("Immagine vuota\n");
		exit(0);
	}

	int i=0,j=0,offset=(imageWidth/2) - 1;

	double time_A=0,time_D=0,time_AA=0,time_DD=0;
	
	initial_time = omp_get_wtime();
	time_A = omp_get_wtime();
    #pragma omp parallel
    {
		#pragma omp for private(i,j) schedule(static)
		for (i = 0; i < imageHeight ; i++) {

			//Matrix A
			new_image.at<double>(i,0) = coefficientsLP[0] * image.at<double>(i,0) 
				+ coefficientsLP[1] * (image.at<double>(i, 1) + ZERO_VALUE)
				+ coefficientsLP[2] * (image.at<double>(i, 2) + ZERO_VALUE)
				+ coefficientsLP[3] * (image.at<double>(i, 3) + ZERO_VALUE)
				+ coefficientsLP[4] * (image.at<double>(i, 4) + ZERO_VALUE);

			new_image.at<double>(i,1) = coefficientsLP[0] * image.at<double>(i, 2) 
				+ coefficientsLP[1] * (image.at<double>(i, 3) + image.at<double>(i, 1))
				+ coefficientsLP[2] * (image.at<double>(i, 4) + image.at<double>(i, 0))
				+ coefficientsLP[3] * (image.at<double>(i, 5) + ZERO_VALUE)
				+ coefficientsLP[4] * (image.at<double>(i, 6) + ZERO_VALUE);

			new_image.at<double>(i,2) = coefficientsLP[0] * image.at<double>(i, 4) 
				+ coefficientsLP[1] * (image.at<double>(i, 5) + image.at<double>(i, 3))
				+ coefficientsLP[2] * (image.at<double>(i, 6) + image.at<double>(i, 2))
				+ coefficientsLP[3] * (image.at<double>(i, 7) + image.at<double>(i, 1))
				+ coefficientsLP[4] * (image.at<double>(i, 8) + image.at<double>(i, 0));
				
				
				
			//Matrix D				
			new_image.at<double>(i,0+offset) = coefficientsHP[0] * image.at<double>(i,0) 
				+ coefficientsHP[1] * (image.at<double>(i, 1) + ZERO_VALUE)
				+ coefficientsHP[2] * (image.at<double>(i, 2) + ZERO_VALUE)
				+ coefficientsHP[3] * (image.at<double>(i, 3) + ZERO_VALUE)
				+ coefficientsHP[4] * (image.at<double>(i, 4) + ZERO_VALUE);

			new_image.at<double>(i,1+offset) = coefficientsHP[0] * image.at<double>(i, 2) 
				+ coefficientsHP[1] * (image.at<double>(i, 3) + image.at<double>(i, 1))
				+ coefficientsHP[2] * (image.at<double>(i, 4) + image.at<double>(i, 0))
				+ coefficientsHP[3] * (image.at<double>(i, 5) + ZERO_VALUE)
				+ coefficientsHP[4] * (image.at<double>(i, 6) + ZERO_VALUE);

			new_image.at<double>(i,2+offset) = coefficientsHP[0] * image.at<double>(i, 4) 
				+ coefficientsHP[1] * (image.at<double>(i, 5) + image.at<double>(i, 3))
				+ coefficientsHP[2] * (image.at<double>(i, 6) + image.at<double>(i, 2))
				+ coefficientsHP[3] * (image.at<double>(i, 7) + image.at<double>(i, 1))
				+ coefficientsHP[4] * (image.at<double>(i, 8) + image.at<double>(i, 0));


			for (j = 3; j < (imageWidth/2) - 2; j++) {
				
				//Matrix A
				new_image.at<double>(i,j) = coefficientsLP[0] * image.at<double>(i, 2*j)
				+ coefficientsLP[1] * (image.at<double>(i, 2*j + 1) + image.at<double>(i, 2*j - 1))
				+ coefficientsLP[2] * (image.at<double>(i, 2*j + 2) + image.at<double>(i, 2*j - 2))
				+ coefficientsLP[3] * (image.at<double>(i, 2*j + 3) + image.at<double>(i, 2*j - 3))
				+ coefficientsLP[4] * (image.at<double>(i, 2*j + 4) + image.at<double>(i, 2*j - 4));
				
				//Matrix D
				new_image.at<double>(i,j+offset) = coefficientsHP[0] * image.at<double>(i, 2*j)
				+ coefficientsHP[1] * (image.at<double>(i, 2*j + 1) + image.at<double>(i, 2*j - 1))
				+ coefficientsHP[2] * (image.at<double>(i, 2*j + 2) + image.at<double>(i, 2*j - 2))
				+ coefficientsHP[3] * (image.at<double>(i, 2*j + 3) + image.at<double>(i, 2*j - 3))
				+ coefficientsHP[4] * (image.at<double>(i, 2*j + 4) + image.at<double>(i, 2*j - 4));
			}
			
			//Matrix A
			new_image.at<double>(i,padding[0]) = coefficientsLP[0] * image.at<double>(i, 2 * padding[0])
				+ coefficientsLP[1] * (image.at<double>(i, 2 * (padding[0])+ 1) 
				+ image.at<double>(i, 2 *(padding[0]) - 1))
				+ coefficientsLP[2] * (image.at<double>(i, 2 *(padding[0]) + 2) 
				+ image.at<double>(i, 2 *(padding[0]) - 2))
				+ coefficientsLP[3] * (image.at<double>(i, 2 *(padding[0]) + 3) 
				+ image.at<double>(i, 2 *(padding[0]) - 3))
				+ coefficientsLP[4] * (ZERO_VALUE + image.at<double>(i, 2 *(padding[0]) - 4));

			new_image.at<double>(i,padding[1])  = coefficientsLP[0] * image.at<double>(i, 2 *(padding[1]))
				+ coefficientsLP[1] * (image.at<double>(i, 2 *(padding[1]) + 1) 
				+ image.at<double>(i, 2 *(padding[1]) - 1))
				+ coefficientsLP[2] * (ZERO_VALUE + image.at<double>(i, 2 *(padding[1]) - 2))
				+ coefficientsLP[3] * (ZERO_VALUE + image.at<double>(i, 2 *(padding[1]) - 3))
				+ coefficientsLP[4] * (ZERO_VALUE + image.at<double>(i, 2 * (padding[1]) - 4));

			new_image.at<double>(i,padding[2]) = ZERO_VALUE
				+ coefficientsLP[1] * (ZERO_VALUE + image.at<double>(i, 2 *(padding[2]) - 1))
				+ coefficientsLP[2] * (ZERO_VALUE + image.at<double>(i, 2 * (padding[2])- 2))
				+ coefficientsLP[3] * (ZERO_VALUE + image.at<double>(i, 2 * (padding[2]) - 3))
				+ coefficientsLP[4] * (ZERO_VALUE + image.at<double>(i, 2 * (padding[2])- 4));
				
				
				
			//Matrix D	
			new_image.at<double>(i,padding[0]+offset) = coefficientsHP[0] * image.at<double>(i, 2 * padding[0])
				+ coefficientsHP[1] * (image.at<double>(i, 2 * (padding[0])+ 1) 
				+ image.at<double>(i, 2 *(padding[0]) - 1))
				+ coefficientsHP[2] * (image.at<double>(i, 2 *(padding[0]) + 2) 
				+ image.at<double>(i, 2 *(padding[0]) - 2))
				+ coefficientsHP[3] * (image.at<double>(i, 2 *(padding[0]) + 3) 
				+ image.at<double>(i, 2 *(padding[0]) - 3))
				+ coefficientsHP[4] * (ZERO_VALUE + image.at<double>(i, 2 *(padding[0]) - 4));

			new_image.at<double>(i,padding[1]+offset)  = coefficientsHP[0] * image.at<double>(i, 2 *(padding[1]))
				+ coefficientsHP[1] * (image.at<double>(i, 2 *(padding[1]) + 1) 
				+ image.at<double>(i, 2 *(padding[1]) - 1))
				+ coefficientsHP[2] * (ZERO_VALUE + image.at<double>(i, 2 *(padding[1]) - 2))
				+ coefficientsHP[3] * (ZERO_VALUE + image.at<double>(i, 2 *(padding[1]) - 3))
				+ coefficientsHP[4] * (ZERO_VALUE + image.at<double>(i, 2 * (padding[1]) - 4));

			new_image.at<double>(i,padding[2]+offset) = ZERO_VALUE
				+ coefficientsHP[1] * (ZERO_VALUE + image.at<double>(i, 2 *(padding[2]) - 1))
				+ coefficientsHP[2] * (ZERO_VALUE + image.at<double>(i, 2 * (padding[2]) - 2))
				+ coefficientsHP[3] * (ZERO_VALUE + image.at<double>(i, 2 * (padding[2]) - 3))
				+ coefficientsHP[4] * (ZERO_VALUE + image.at<double>(i, 2 * (padding[2]) - 4));
		}
    }	
    
    		time_D = omp_get_wtime();
    		time_D -= time_A; // Execution time of matrices A and D
    
		padding[0] = (imageHeight/ 2) - 2;
		padding[1] = (imageHeight/ 2) - 1;
		padding[2] = (imageHeight/ 2);
		
		offset = (imageHeight/2) - 1;
		
		time_AA = omp_get_wtime();
        #pragma omp parallel
        {
		#pragma omp for private(i,j) schedule(static)
		for (i = 0; i < imageWidth ; i++) {
			
			//Matrices AA and AD
			tmp_image_1.at<double>(0,i) = coefficientsLP[0] * new_image.at<double>(0,i) 
				+ coefficientsLP[1] * (new_image.at<double>(1,i) + ZERO_VALUE)
				+ coefficientsLP[2] * (new_image.at<double>(2,i) + ZERO_VALUE)
				+ coefficientsLP[3] * (new_image.at<double>(3,i) + ZERO_VALUE)
				+ coefficientsLP[4] * (new_image.at<double>(4,i) + ZERO_VALUE);

			tmp_image_1.at<double>(1,i) = coefficientsLP[0] * new_image.at<double>(2,i) 
				+ coefficientsLP[1] * (new_image.at<double>(3,i) + new_image.at<double>(1,i))
				+ coefficientsLP[2] * (new_image.at<double>(4,i) + new_image.at<double>(0,i))
				+ coefficientsLP[3] * (new_image.at<double>(5,i) + ZERO_VALUE)
				+ coefficientsLP[4] * (new_image.at<double>(6,i) + ZERO_VALUE);

			tmp_image_1.at<double>(2,i) = coefficientsLP[0] * new_image.at<double>(4,i) 
				+ coefficientsLP[1] * (new_image.at<double>(5,i) + new_image.at<double>(3,i))
				+ coefficientsLP[2] * (new_image.at<double>(6,i) + new_image.at<double>(2,i))
				+ coefficientsLP[3] * (new_image.at<double>(7,i) + new_image.at<double>(1,i))
				+ coefficientsLP[4] * (new_image.at<double>(8,i) + new_image.at<double>(0,i));

			//Matrices DA and DD
			tmp_image_2.at<double>(0+offset,i) = coefficientsHP[0] * new_image.at<double>(0,i) 
				+ coefficientsHP[1] * (new_image.at<double>(1,i) + ZERO_VALUE)
				+ coefficientsHP[2] * (new_image.at<double>(2,i) + ZERO_VALUE)
				+ coefficientsHP[3] * (new_image.at<double>(3,i) + ZERO_VALUE)
				+ coefficientsHP[4] * (new_image.at<double>(4,i) + ZERO_VALUE);

			tmp_image_2.at<double>(1+offset,i) = coefficientsHP[0] * new_image.at<double>(2,i) 
				+ coefficientsHP[1] * (new_image.at<double>(3,i) + new_image.at<double>(1,i))
				+ coefficientsHP[2] * (new_image.at<double>(4,i) + new_image.at<double>(0,i))
				+ coefficientsHP[3] * (new_image.at<double>(5,i) + ZERO_VALUE)
				+ coefficientsHP[4] * (new_image.at<double>(6,i) + ZERO_VALUE);

			tmp_image_2.at<double>(2+offset,i) = coefficientsHP[0] * new_image.at<double>(4,i) 
				+ coefficientsHP[1] * (new_image.at<double>(5,i) + new_image.at<double>(3,i))
				+ coefficientsHP[2] * (new_image.at<double>(6,i) + new_image.at<double>(2,i))
				+ coefficientsHP[3] * (new_image.at<double>(7,i) + new_image.at<double>(1,i))
				+ coefficientsHP[4] * (new_image.at<double>(8,i) + new_image.at<double>(0,i));


			for (j = 3; j < (imageHeight/2) - 2; j++) {
				//Matrices AA and AD
				tmp_image_1.at<double>(j,i) = coefficientsLP[0] * new_image.at<double>(2*j,i)
					+ coefficientsLP[1] * (new_image.at<double>(2*j + 1,i) + new_image.at<double>(2*j - 1,i))
					+ coefficientsLP[2] * (new_image.at<double>(2*j + 2,i) + new_image.at<double>(2*j - 2,i))
					+ coefficientsLP[3] * (new_image.at<double>(2*j + 3,i) + new_image.at<double>(2*j - 3,i))
					+ coefficientsLP[4] * (new_image.at<double>(2*j + 4,i) + new_image.at<double>(2*j - 4,i));
					
				//Matrices DA and DD	
				tmp_image_2.at<double>(j+offset,i) = coefficientsHP[0] * new_image.at<double>(2*j,i)
					+ coefficientsHP[1] * (new_image.at<double>(2*j + 1,i) + new_image.at<double>(2*j - 1,i))
					+ coefficientsHP[2] * (new_image.at<double>(2*j + 2,i) + new_image.at<double>(2*j - 2,i))
					+ coefficientsHP[3] * (new_image.at<double>(2*j + 3,i) + new_image.at<double>(2*j - 3,i))
					+ coefficientsHP[4] * (new_image.at<double>(2*j + 4,i) + new_image.at<double>(2*j - 4,i));
			}
			
			//Matrices AA and AD
			tmp_image_1.at<double>(padding[0],i) = coefficientsLP[0] * new_image.at<double>(2 * padding[0],i)
				+ coefficientsLP[1] * (new_image.at<double>(2 * (padding[0])+ 1,i) 
				+ new_image.at<double>(2 *(padding[0]) - 1,i))
				+ coefficientsLP[2] * (new_image.at<double>(2 *(padding[0]) + 2,i) 
				+ new_image.at<double>(2 *(padding[0]) - 2,i))
				+ coefficientsLP[3] * (new_image.at<double>(2 *(padding[0]) + 3,i) 
				+ new_image.at<double>(2 *(padding[0]) - 3,i))
				+ coefficientsLP[4] * (ZERO_VALUE + new_image.at<double>(2 *(padding[0]) - 4,i));

			tmp_image_1.at<double>(padding[1],i) = coefficientsLP[0] * new_image.at<double>(2*(padding[1]),i)
				+ coefficientsLP[1] * (new_image.at<double>(2 *(padding[1]) + 1,i) + new_image.at<double>(2 *(padding[1]) - 1,i))
				+ coefficientsLP[2] * (ZERO_VALUE + new_image.at<double>(2 *(padding[1]) - 2,i))
				+ coefficientsLP[3] * (ZERO_VALUE + new_image.at<double>(2 *(padding[1]) - 3,i))
				+ coefficientsLP[4] * (ZERO_VALUE + new_image.at<double>(2 * (padding[1]) - 4,i));

			tmp_image_1.at<double>(padding[2],i) = ZERO_VALUE
				+ coefficientsLP[1] * (ZERO_VALUE + new_image.at<double>(2 *(padding[2]) - 1,i))
				+ coefficientsLP[2] * (ZERO_VALUE + new_image.at<double>(2 * (padding[2])- 2,i))
				+ coefficientsLP[3] * (ZERO_VALUE + new_image.at<double>(2 * (padding[2]) - 3,i))
				+ coefficientsLP[4] * (ZERO_VALUE + new_image.at<double>(2 * (padding[2])- 4,i));
				
			//Matrices DA and DD
			tmp_image_2.at<double>(padding[0]+offset,i) = coefficientsHP[0] * new_image.at<double>(2 * padding[0],i)
				+ coefficientsHP[1] * (new_image.at<double>(2 * (padding[0])+ 1,i) 
				+ new_image.at<double>(2 *(padding[0]) - 1,i))
				+ coefficientsHP[2] * (new_image.at<double>(2 *(padding[0]) + 2,i) 
				+ new_image.at<double>(2 *(padding[0]) - 2,i))
				+ coefficientsHP[3] * (new_image.at<double>(2 *(padding[0]) + 3,i) 
				+ new_image.at<double>(2 *(padding[0]) - 3,i))
				+ coefficientsHP[4] * (ZERO_VALUE + new_image.at<double>(2 *(padding[0]) - 4,i));

			tmp_image_2.at<double>(padding[1]+offset,i) = coefficientsHP[0] * new_image.at<double>(2*(padding[1]),i)
				+ coefficientsHP[1] * (new_image.at<double>(2 *(padding[1]) + 1,i) + new_image.at<double>(2 *(padding[1]) - 1,i))
				+ coefficientsHP[2] * (ZERO_VALUE + new_image.at<double>(2 *(padding[1]) - 2,i))
				+ coefficientsHP[3] * (ZERO_VALUE + new_image.at<double>(2 *(padding[1]) - 3,i))
				+ coefficientsHP[4] * (ZERO_VALUE + new_image.at<double>(2 * (padding[1]) - 4,i));

			tmp_image_2.at<double>(padding[2]+offset,i) = ZERO_VALUE
				+ coefficientsHP[1] * (ZERO_VALUE + new_image.at<double>(2 *(padding[2]) - 1,i))
				+ coefficientsHP[2] * (ZERO_VALUE + new_image.at<double>(2 * (padding[2])- 2,i))
				+ coefficientsHP[3] * (ZERO_VALUE + new_image.at<double>(2 * (padding[2]) - 3,i))
				+ coefficientsHP[4] * (ZERO_VALUE + new_image.at<double>(2 * (padding[2])- 4,i));
			}
			
        }
        time_DD = omp_get_wtime();
        time_DD -= time_AA; // Execution time of matrices AA, AD,DA and DD
        	
	final_time = omp_get_wtime();
	final_time -= initial_time;	
	printf("time %lf\n", final_time);
	
	printf("time A e D %lf\n", time_D);
	printf("time AA e DD %lf\n", time_DD);

//   Copying of tmp_image_1 and tmp_image_2 in new_image
    for(int i=0;i<imageWidth;i++){
        for(int j=0;j<imageHeight/2;j++){
            new_image.at<double>(j,i) = tmp_image_1.at<double>(j,i);
        }
        for(int j=imageHeight/2;j<imageHeight;j++){
            new_image.at<double>(j,i) = tmp_image_2.at<double>(j,i);
        }
    }
    
    
    	cv::Mat final_image(imageHeight, imageWidth, CV_8UC1);
    	
    	cv::normalize(new_image, final_image, 0,255,NORM_MINMAX, CV_8UC1);
    	
    
    	imshow("Parallel CDF 9/7", final_image);
    	waitKey(0);
    	
/*	vector<int> compression_params;

	compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);

	compression_params.push_back(60);


    printf("%s %d %d\n",file_name, name_length, pos);

	imwrite(file_name,new_image,compression_params);

	waitKey(0);*/

	return 0;
}
