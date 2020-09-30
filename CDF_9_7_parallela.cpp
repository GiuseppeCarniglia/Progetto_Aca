#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <math.h>
#include <omp.h>

using namespace cv;
using namespace std;

#define ZERO_VALUE (uchar)0
#define CHUNK_SIZE (8)

/*
   
   
   ________________________________________________9/7-CDF Transform_________________________________________________________
   * 
   * 
  
 */



void wavelet_row(Mat src, Mat dst, double coefficients[],int offset);

void wavelet_cols(Mat src,Mat dst, double coefficients[],int offset);

int main()
{
	const char* imgName = "./immagini/volto.jpg";
	Mat image = imread(imgName, IMREAD_GRAYSCALE);
	int imageWidth = image.size().width;
	int imageHeight = image.size().height;

	Mat new_image = cv::Mat(imageHeight,imageWidth,CV_8UC1);

	Mat tmp_image_1 = cv::Mat(imageHeight,imageWidth,CV_8UC1);
	Mat tmp_image_2 = cv::Mat(imageHeight,imageWidth,CV_8UC1);

	double coefficientsLP[5] = { 0.602949018236,0.266864118443,-0.078223266529,-0.016864118443,0.26748757411 };
	double coefficientsHP[5] = { 1.11508705, -0.59127176314, -0.057543526229, 0.091271763114, 0.0 };

	double initial_time = 0, final_time = 0;

	initial_time = omp_get_wtime();
	
	omp_set_num_threads(omp_get_max_threads());
	#pragma omp parallel 
	{
		#pragma omp sections 
		{
			#pragma omp section 
			{
				wavelet_row(image,new_image,coefficientsLP,0);
			}	
				
			#pragma omp section 
			{
				wavelet_row(image,new_image,coefficientsHP,(imageWidth/2));
			}
		}// end sections

		#pragma omp sections
		{
			#pragma omp section
			{
				wavelet_cols(new_image,tmp_image_1,coefficientsLP,0);
			}
			#pragma omp section
			{
				wavelet_cols(new_image,tmp_image_2,coefficientsHP,(imageHeight/2));
			}
		}
	}//end parallel
			
	final_time = omp_get_wtime();
	final_time -= initial_time;	
	printf("time %lf\n", final_time);


	tmp_image_1.copyTo(new_image(Rect(0,0,imageWidth,imageHeight/2)));	
	tmp_image_2.copyTo(new_image(Rect(0,imageHeight/2,imageWidth,imageHeight)));	
	
	vector<int> compression_params;

	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);

	compression_params.push_back(60);

	imwrite("./immagini_modificate/CDF_9_7_parallela/CDF_9_7_parallela.jpg",new_image,compression_params);

	waitKey(0);

	return 0;
}




void wavelet_row(Mat src, Mat dst, double coefficients[],int offset){
	
	int srcWidth = src.size().width;
	int srcHeight = src.size().height;
	
	int padding[3] = {(srcWidth/2) - 2 ,(srcWidth/2) - 1 ,(srcWidth/2)};
	
	int i=0,j=0;

	#pragma omp parallel for private(i,j) schedule(static)
	for (i = 0; i < srcHeight ; i++) {
					
		dst.at<uchar>(i,0 + offset) = coefficients[0] * src.at<uchar>(i,0) 
			+ coefficients[1] * (src.at<uchar>(i, 1) + ZERO_VALUE)
			+ coefficients[2] * (src.at<uchar>(i, 2) + ZERO_VALUE)
			+ coefficients[3] * (src.at<uchar>(i, 3) + ZERO_VALUE)
			+ coefficients[4] * (src.at<uchar>(i, 4) + ZERO_VALUE);

		dst.at<uchar>(i,1 + offset) = coefficients[0] * src.at<uchar>(i, 2) 
			+ coefficients[1] * (src.at<uchar>(i, 3) + src.at<uchar>(i, 1))
			+ coefficients[2] * (src.at<uchar>(i, 4) + src.at<uchar>(i, 0))
			+ coefficients[3] * (src.at<uchar>(i, 5) + ZERO_VALUE)
			+ coefficients[4] * (src.at<uchar>(i, 6) + ZERO_VALUE);

		dst.at<uchar>(i,2 + offset) = coefficients[0] * src.at<uchar>(i, 4) 
			+ coefficients[1] * (src.at<uchar>(i, 5) + src.at<uchar>(i, 3))
			+ coefficients[2] * (src.at<uchar>(i, 6) + src.at<uchar>(i, 2))
			+ coefficients[3] * (src.at<uchar>(i, 7) + src.at<uchar>(i, 1))
			+ coefficients[4] * (src.at<uchar>(i, 8) + src.at<uchar>(i, 0));


		for (j = 3; j < (srcWidth/2) - 2; j++) {

			dst.at<uchar>(i,j + offset) = coefficients[0] * src.at<uchar>(i, 2*j)
				+ coefficients[1] * (src.at<uchar>(i, 2*j + 1) + src.at<uchar>(i, 2*j - 1))
				+ coefficients[2] * (src.at<uchar>(i, 2*j + 2) + src.at<uchar>(i, 2*j - 2))
				+ coefficients[3] * (src.at<uchar>(i, 2*j + 3) + src.at<uchar>(i, 2*j - 3))
				+ coefficients[4] * (src.at<uchar>(i, 2*j + 4) + src.at<uchar>(i, 2*j - 4));
		}

		dst.at<uchar>(i,padding[0] + offset) = coefficients[0] * src.at<uchar>(i, 2 * padding[0])
			+ coefficients[1] * (src.at<uchar>(i, 2 * (padding[0])+ 1) 
			+ src.at<uchar>(i, 2 *(padding[0]) - 1))
			+ coefficients[2] * (src.at<uchar>(i, 2 *(padding[0]) + 2) 
			+ src.at<uchar>(i, 2 *(padding[0]) - 2))
			+ coefficients[3] * (src.at<uchar>(i, 2 *(padding[0]) + 3) 
			+ src.at<uchar>(i, 2 *(padding[0]) - 3))
			+ coefficients[4] * (ZERO_VALUE + src.at<uchar>(i, 2 *(padding[0]) - 4));

		dst.at<uchar>(i,padding[1] + offset) = coefficients[0] * src.at<uchar>(i, 2 *(padding[1]))
			+ coefficients[1] * (src.at<uchar>(i, 2 *(padding[1]) + 1) 
			+ src.at<uchar>(i, 2 *(padding[1]) - 1))
			+ coefficients[2] * (ZERO_VALUE + src.at<uchar>(i, 2 *(padding[1]) - 2))
			+ coefficients[3] * (ZERO_VALUE + src.at<uchar>(i, 2 *(padding[1]) - 3))
			+ coefficients[4] * (ZERO_VALUE + src.at<uchar>(i, 2 * (padding[1]) - 4));

		dst.at<uchar>(i,padding[2] + offset) = ZERO_VALUE
			+ coefficients[1] * (ZERO_VALUE + src.at<uchar>(i, 2 *(padding[2]) - 1))
			+ coefficients[2] * (ZERO_VALUE + src.at<uchar>(i, 2 * (padding[2])- 2))
			+ coefficients[3] * (ZERO_VALUE + src.at<uchar>(i, 2 * (padding[2]) - 3))
			+ coefficients[4] * (ZERO_VALUE + src.at<uchar>(i, 2 * (padding[2])- 4));
	}
}


void wavelet_cols(Mat src,Mat dst, double coefficients[],int offset){

	int srcWidth = src.size().width;
	int srcHeight = src.size().height;
	
	int padding[3] = {(srcWidth / 2) - 2 ,(srcWidth / 2) - 1 ,(srcWidth / 2)};
	
	int i=0,j=0;
	
	#pragma omp parallel for private(i,j) schedule(static)
	for (i = 0; i < srcWidth ; i++) {
			
		dst.at<uchar>(0+offset,i) = coefficients[0] * src.at<uchar>(0,i) 
			+ coefficients[1] * (src.at<uchar>(1,i) + ZERO_VALUE)
			+ coefficients[2] * (src.at<uchar>(2,i) + ZERO_VALUE)
			+ coefficients[3] * (src.at<uchar>(3,i) + ZERO_VALUE)
			+ coefficients[4] * (src.at<uchar>(4,i) + ZERO_VALUE);

		dst.at<uchar>(1+offset,i) = coefficients[0] * src.at<uchar>(2,i) 
			+ coefficients[1] * (src.at<uchar>(3,i) + src.at<uchar>(1,i))
			+ coefficients[2] * (src.at<uchar>(4,i) + src.at<uchar>(0,i))
			+ coefficients[3] * (src.at<uchar>(5,i) + ZERO_VALUE)
			+ coefficients[4] * (src.at<uchar>(6,i) + ZERO_VALUE);

		dst.at<uchar>(2+offset,i) = coefficients[0] * src.at<uchar>(4,i) 
			+ coefficients[1] * (src.at<uchar>(5,i) + src.at<uchar>(3,i))
			+ coefficients[2] * (src.at<uchar>(6,i) + src.at<uchar>(2,i))
			+ coefficients[3] * (src.at<uchar>(7,i) + src.at<uchar>(1,i))
			+ coefficients[4] * (src.at<uchar>(8,i) + src.at<uchar>(0,i));


		for (j = 3; j < (srcHeight/2) - 2; j++) {

			dst.at<uchar>(j+offset,i) = coefficients[0] * src.at<uchar>(2*j,i)
				+ coefficients[1] * (src.at<uchar>(2*j + 1,i) + src.at<uchar>(2*j - 1,i))
				+ coefficients[2] * (src.at<uchar>(2*j + 2,i) + src.at<uchar>(2*j - 2,i))
				+ coefficients[3] * (src.at<uchar>(2*j + 3,i) + src.at<uchar>(2*j - 3,i))
				+ coefficients[4] * (src.at<uchar>(2*j + 4,i) + src.at<uchar>(2*j - 4,i));
		}

		dst.at<uchar>(padding[0]+offset,i) = coefficients[0] * src.at<uchar>(2 * padding[0],i)
			+ coefficients[1] * (src.at<uchar>(2 * (padding[0])+ 1,i) 
			+ src.at<uchar>(2 *(padding[0]) - 1),i)
			+ coefficients[2] * (src.at<uchar>(2 *(padding[0]) + 2,i) 
			+ src.at<uchar>(2 *(padding[0]) - 2),i)
			+ coefficients[3] * (src.at<uchar>(2 *(padding[0]) + 3,i) 
			+ src.at<uchar>(2 *(padding[0]) - 3),i)
			+ coefficients[4] * (ZERO_VALUE + src.at<uchar>(2 *(padding[0]) - 4),i);

		dst.at<uchar>(padding[1]+offset,i) = coefficients[0] * src.at<uchar>(2*(padding[1]),i)
			+ coefficients[1] * (src.at<uchar>(2 *(padding[1]) + 1,i) 
			+ src.at<uchar>(2 *(padding[1]) - 1),i)
			+ coefficients[2] * (ZERO_VALUE + src.at<uchar>(2 *(padding[1]) - 2),i)
			+ coefficients[3] * (ZERO_VALUE + src.at<uchar>(2 *(padding[1]) - 3),i)
			+ coefficients[4] * (ZERO_VALUE + src.at<uchar>(2 * (padding[1]) - 4),i);

		dst.at<uchar>(padding[2]+offset,i) = ZERO_VALUE
			+ coefficients[1] * (ZERO_VALUE + src.at<uchar>(2 *(padding[2]) - 1),i)
			+ coefficients[2] * (ZERO_VALUE + src.at<uchar>(2 * (padding[2])- 2),i)
			+ coefficients[3] * (ZERO_VALUE + src.at<uchar>(2 * (padding[2]) - 3),i)
			+ coefficients[4] * (ZERO_VALUE + src.at<uchar>(2 * (padding[2])- 4),i);
		}
}

























