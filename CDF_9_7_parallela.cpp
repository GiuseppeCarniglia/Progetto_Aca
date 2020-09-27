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



void wavelet_row(Mat src, Mat dst, uchar *a, double coefficients[],int offset);

void wavelet_cols(Mat src,Mat dst, uchar *a, double coefficients[],int offset);

int main()
{
	const char* imgName = "./immagini/volto.jpg";
	Mat image = imread(imgName, IMREAD_GRAYSCALE);
	Mat new_image = cv::Mat(image.size().height,image.size().width,CV_8UC1);
	int imageWidth = image.size().width;
	int imageHeight = image.size().height;
	int w = 0;
	uchar *a;
	a = (uchar*)malloc(imageWidth*imageHeight * sizeof(uchar));
	
	uchar *d;
	d = (uchar*)malloc(imageWidth*imageHeight * sizeof(uchar));
	double coefficientsLP[5] = { 0.602949018236,0.266864118443,-0.078223266529,-0.016864118443,0.26748757411 };
	double coefficientsHP[5] = { 1.11508705, -0.59127176314, -0.057543526229, 0.091271763114, 0.0 };
	int padding[3] = { (w / 2) - 2 ,(w / 2) - 1 ,(w / 2) };
	double initial_time = 0, final_time = 0;

	initial_time = omp_get_wtime();
	
	w = imageWidth;
	
	int i=0, j=0,k=0;
	
		
	omp_set_num_threads(2);
	#pragma omp parallel private(i,j)
	{
		#pragma omp sections 
		{
			#pragma omp section 
			{
				wavelet_row(image,new_image,a,coefficientsLP,0);
			}	
				
			#pragma omp section 
			{
				wavelet_row(image,new_image,d,coefficientsHP,(imageWidth/2));
			}
		}// end sections
	#pragma omp barrier

		#pragma omp sections
		{
			#pragma omp section
			{
				wavelet_cols(image,new_image,a,coefficientsLP,0);
			}
			

		}


	
	}//end parallel
			
			
	free(a);
	free(d);
	final_time = omp_get_wtime();
	final_time -= initial_time;
	
	printf("time %lf\n", final_time);


	vector<int> compression_params;

	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);

	compression_params.push_back(60);

	imwrite("./immagini/immagini_modificate/CDF_9_7_parallela/CDF_9_7_parallela.jpg",new_image,compression_params);

	waitKey(0);

	return 0;
}




void wavelet_row(Mat src, Mat dst, uchar *a, double coefficients[],int offset){
	
	int srcWidth = src.size().width;
	int srcHeight = src.size().height;
	
	int padding[3] = {(srcWidth / 2) - 2 ,(srcWidth / 2) - 1 ,(srcWidth / 2)};
	
	int i=0,j=0,k=0;
	

	//#pragma omp for schedule(dynamic,CHUNK_SIZE) private(i,j,k)

	for (i = 0; i < srcHeight ; i++) {
					
		//Calcolo matrice di approssimazione a
		a[0] = coefficients[0] * src.at<uchar>(i,0) 
			+ coefficients[1] * (src.at<uchar>(i, 1) + ZERO_VALUE)
			+ coefficients[2] * (src.at<uchar>(i, 2) + ZERO_VALUE)
			+ coefficients[3] * (src.at<uchar>(i, 3) + ZERO_VALUE)
			+ coefficients[4] * (src.at<uchar>(i, 4) + ZERO_VALUE);

		a[1] = coefficients[0] * src.at<uchar>(i, 2) 
			+ coefficients[1] * (src.at<uchar>(i, 3) + src.at<uchar>(i, 1))
			+ coefficients[2] * (src.at<uchar>(i, 4) + src.at<uchar>(i, 0))
			+ coefficients[3] * (src.at<uchar>(i, 5) + ZERO_VALUE)
			+ coefficients[4] * (src.at<uchar>(i, 6) + ZERO_VALUE);

		a[2] = coefficients[0] * src.at<uchar>(i, 4) 
			+ coefficients[1] * (src.at<uchar>(i, 5) + src.at<uchar>(i, 3))
			+ coefficients[2] * (src.at<uchar>(i, 6) + src.at<uchar>(i, 2))
			+ coefficients[3] * (src.at<uchar>(i, 7) + src.at<uchar>(i, 1))
			+ coefficients[4] * (src.at<uchar>(i, 8) + src.at<uchar>(i, 0));


		for (j = 3; j < (srcWidth/2) - 2; j++) {

			a[j] = coefficients[0] * src.at<uchar>(i, 2*j)
				+ coefficients[1] * (src.at<uchar>(i, 2*j + 1) + src.at<uchar>(i, 2*j - 1))
				+ coefficients[2] * (src.at<uchar>(i, 2*j + 2) + src.at<uchar>(i, 2*j - 2))
				+ coefficients[3] * (src.at<uchar>(i, 2*j + 3) + src.at<uchar>(i, 2*j - 3))
				+ coefficients[4] * (src.at<uchar>(i, 2*j + 4) + src.at<uchar>(i, 2*j - 4));
		}

		a[padding[0]] = coefficients[0] * src.at<uchar>(i, 2 * padding[0])
			+ coefficients[1] * (src.at<uchar>(i, 2 * (padding[0])+ 1) 
			+ src.at<uchar>(i, 2 *(padding[0]) - 1))
			+ coefficients[2] * (src.at<uchar>(i, 2 *(padding[0]) + 2) 
			+ src.at<uchar>(i, 2 *(padding[0]) - 2))
			+ coefficients[3] * (src.at<uchar>(i, 2 *(padding[0]) + 3) 
			+ src.at<uchar>(i, 2 *(padding[0]) - 3))
			+ coefficients[4] * (ZERO_VALUE + src.at<uchar>(i, 2 *(padding[0]) - 4));

		a[padding[1]] = coefficients[0] * src.at<uchar>(i, 2 *(padding[1]))
			+ coefficients[1] * (src.at<uchar>(i, 2 *(padding[1]) + 1) 
			+ src.at<uchar>(i, 2 *(padding[1]) - 1))
			+ coefficients[2] * (ZERO_VALUE + src.at<uchar>(i, 2 *(padding[1]) - 2))
			+ coefficients[3] * (ZERO_VALUE + src.at<uchar>(i, 2 *(padding[1]) - 3))
			+ coefficients[4] * (ZERO_VALUE + src.at<uchar>(i, 2 * (padding[1]) - 4));

		a[padding[2]] = ZERO_VALUE
			+ coefficients[1] * (ZERO_VALUE + src.at<uchar>(i, 2 *(padding[2]) - 1))
			+ coefficients[2] * (ZERO_VALUE + src.at<uchar>(i, 2 * (padding[2])- 2))
			+ coefficients[3] * (ZERO_VALUE + src.at<uchar>(i, 2 * (padding[2]) - 3))
			+ coefficients[4] * (ZERO_VALUE + src.at<uchar>(i, 2 * (padding[2])- 4));
		
		for(k = 0; k < (srcWidth/2) ; k++) {
				dst.at<uchar>(i,k+offset) = a[k];
			}
		}

}


void wavelet_cols(Mat src,Mat dst, uchar *a, double coefficients[],int offset){

	

	// TODO: Similar function body as above only on columns
	int srcWidth = src.size().width;
	int srcHeight = src.size().height;
	
	int padding[3] = {(srcWidth / 2) - 2 ,(srcWidth / 2) - 1 ,(srcWidth / 2)};
	
	int i=0,j=0,k=0;
	

	//#pragma omp for schedule(dynamic,CHUNK_SIZE) private(i,j,k)

	for (i = 0; i < srcWidth ; i++) {
					
		//Calcolo matrice di approssimazione a
		a[0] = coefficients[0] * src.at<uchar>(0,i) 
			+ coefficients[1] * (src.at<uchar>(1,i) + ZERO_VALUE)
			+ coefficients[2] * (src.at<uchar>(2,i) + ZERO_VALUE)
			+ coefficients[3] * (src.at<uchar>(3,i) + ZERO_VALUE)
			+ coefficients[4] * (src.at<uchar>(4,i) + ZERO_VALUE);

		a[1] = coefficients[0] * src.at<uchar>(2,i) 
			+ coefficients[1] * (src.at<uchar>(3,i) + src.at<uchar>(1,i))
			+ coefficients[2] * (src.at<uchar>(4,i) + src.at<uchar>(0,i))
			+ coefficients[3] * (src.at<uchar>(5,i) + ZERO_VALUE)
			+ coefficients[4] * (src.at<uchar>(6,i) + ZERO_VALUE);

		a[2] = coefficients[0] * src.at<uchar>(4,i) 
			+ coefficients[1] * (src.at<uchar>(5,i) + src.at<uchar>(3,i))
			+ coefficients[2] * (src.at<uchar>(6,i) + src.at<uchar>(2,i))
			+ coefficients[3] * (src.at<uchar>(7,i) + src.at<uchar>(1,i))
			+ coefficients[4] * (src.at<uchar>(8,i) + src.at<uchar>(0,i));


		for (j = 3; j < (srcHeight/2) - 2; j++) {

			a[j] = coefficients[0] * src.at<uchar>(2*j,i)
				+ coefficients[1] * (src.at<uchar>(2*j + 1,i) + src.at<uchar>(2*j - 1,i))
				+ coefficients[2] * (src.at<uchar>(2*j + 2,i) + src.at<uchar>(2*j - 2,i))
				+ coefficients[3] * (src.at<uchar>(2*j + 3,i) + src.at<uchar>(2*j - 3,i))
				+ coefficients[4] * (src.at<uchar>(2*j + 4,i) + src.at<uchar>(2*j - 4,i));
		}

		a[padding[0]] = coefficients[0] * src.at<uchar>(2 * padding[0],i)
			+ coefficients[1] * (src.at<uchar>(2 * (padding[0])+ 1,i) 
			+ src.at<uchar>(2 *(padding[0]) - 1),i)
			+ coefficients[2] * (src.at<uchar>(2 *(padding[0]) + 2,i) 
			+ src.at<uchar>(2 *(padding[0]) - 2),i)
			+ coefficients[3] * (src.at<uchar>(2 *(padding[0]) + 3,i) 
			+ src.at<uchar>(2 *(padding[0]) - 3),i)
			+ coefficients[4] * (ZERO_VALUE + src.at<uchar>(2 *(padding[0]) - 4),i);

		a[padding[1]] = coefficients[0] * src.at<uchar>(2*(padding[1]),i)
			+ coefficients[1] * (src.at<uchar>(2 *(padding[1]) + 1,i) 
			+ src.at<uchar>(2 *(padding[1]) - 1),i)
			+ coefficients[2] * (ZERO_VALUE + src.at<uchar>(2 *(padding[1]) - 2),i)
			+ coefficients[3] * (ZERO_VALUE + src.at<uchar>(2 *(padding[1]) - 3),i)
			+ coefficients[4] * (ZERO_VALUE + src.at<uchar>(2 * (padding[1]) - 4),i);

		a[padding[2]] = ZERO_VALUE
			+ coefficients[1] * (ZERO_VALUE + src.at<uchar>(2 *(padding[2]) - 1),i)
			+ coefficients[2] * (ZERO_VALUE + src.at<uchar>(2 * (padding[2])- 2),i)
			+ coefficients[3] * (ZERO_VALUE + src.at<uchar>(2 * (padding[2]) - 3),i)
			+ coefficients[4] * (ZERO_VALUE + src.at<uchar>(2 * (padding[2])- 4),i);
		
		for(k = 0; k < (srcHeight/2) ; k++) {
				dst.at<uchar>(k+offset,i) = a[k];
			}
		}


}

























