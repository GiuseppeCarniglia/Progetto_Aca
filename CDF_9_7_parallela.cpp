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



uchar wavelet_row(Mat src,int i, int j,double coefficients[]);

//void wavelet_cols(Mat src,Mat dst, double coefficients[],int offset);


int main(int argc, char* argv[])
{
	const char* imgName = argv[1];
	Mat image = imread(imgName, IMREAD_GRAYSCALE);
	int imageWidth = image.size().width;
	int imageHeight = image.size().height;

	Mat new_image = cv::Mat(imageHeight,imageWidth,CV_8UC1);

	Mat tmp_image_1 = cv::Mat(imageHeight,imageWidth,CV_8UC1);
	Mat tmp_image_2 = cv::Mat(imageHeight,imageWidth,CV_8UC1);

	double coefficientsLP[5] = { 0.602949018236,0.266864118443,-0.078223266529,-0.016864118443,0.26748757411 };
	double coefficientsHP[5] = { 1.11508705, -0.59127176314, -0.057543526229, 0.091271763114, 0.0 };

	int padding[3] = {(imageWidth/2) - 2 ,(imageWidth/2) - 1 ,(imageWidth/2)};


	double initial_time = 0, final_time = 0;
	
	if(image.empty()){
		perror("Immagine vuota\n");
		exit(0);
	}

	int i=0,j=0,offset=(imageWidth/2) - 1;


	initial_time = omp_get_wtime();
	
		#pragma omp for private(i,j) schedule(static)
		for (i = 0; i < imageHeight ; i++) {
						
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
				
				
				
								
			new_image.at<uchar>(i,0+offset) = coefficientsHP[0] * image.at<uchar>(i,0) 
				+ coefficientsHP[1] * (image.at<uchar>(i, 1) + ZERO_VALUE)
				+ coefficientsHP[2] * (image.at<uchar>(i, 2) + ZERO_VALUE)
				+ coefficientsHP[3] * (image.at<uchar>(i, 3) + ZERO_VALUE)
				+ coefficientsHP[4] * (image.at<uchar>(i, 4) + ZERO_VALUE);

			new_image.at<uchar>(i,1+offset) = coefficientsHP[0] * image.at<uchar>(i, 2) 
				+ coefficientsHP[1] * (image.at<uchar>(i, 3) + image.at<uchar>(i, 1))
				+ coefficientsHP[2] * (image.at<uchar>(i, 4) + image.at<uchar>(i, 0))
				+ coefficientsHP[3] * (image.at<uchar>(i, 5) + ZERO_VALUE)
				+ coefficientsHP[4] * (image.at<uchar>(i, 6) + ZERO_VALUE);

			new_image.at<uchar>(i,2+offset) = coefficientsHP[0] * image.at<uchar>(i, 4) 
				+ coefficientsHP[1] * (image.at<uchar>(i, 5) + image.at<uchar>(i, 3))
				+ coefficientsHP[2] * (image.at<uchar>(i, 6) + image.at<uchar>(i, 2))
				+ coefficientsHP[3] * (image.at<uchar>(i, 7) + image.at<uchar>(i, 1))
				+ coefficientsHP[4] * (image.at<uchar>(i, 8) + image.at<uchar>(i, 0));


			for (j = 3; j < (imageWidth/2) - 2; j++) {

				new_image.at<uchar>(i,j) = coefficientsLP[0] * image.at<uchar>(i, 2*j)
				+ coefficientsLP[1] * (image.at<uchar>(i, 2*j + 1) + image.at<uchar>(i, 2*j - 1))
				+ coefficientsLP[2] * (image.at<uchar>(i, 2*j + 2) + image.at<uchar>(i, 2*j - 2))
				+ coefficientsLP[3] * (image.at<uchar>(i, 2*j + 3) + image.at<uchar>(i, 2*j - 3))
				+ coefficientsLP[4] * (image.at<uchar>(i, 2*j + 4) + image.at<uchar>(i, 2*j - 4));
				
				
				new_image.at<uchar>(i,j+offset) = coefficientsHP[0] * image.at<uchar>(i, 2*j)
				+ coefficientsHP[1] * (image.at<uchar>(i, 2*j + 1) + image.at<uchar>(i, 2*j - 1))
				+ coefficientsHP[2] * (image.at<uchar>(i, 2*j + 2) + image.at<uchar>(i, 2*j - 2))
				+ coefficientsHP[3] * (image.at<uchar>(i, 2*j + 3) + image.at<uchar>(i, 2*j - 3))
				+ coefficientsHP[4] * (image.at<uchar>(i, 2*j + 4) + image.at<uchar>(i, 2*j - 4));
			}

			new_image.at<uchar>(i,padding[0]) = coefficientsLP[0] * image.at<uchar>(i, 2 * padding[0])
				+ coefficientsLP[1] * (image.at<uchar>(i, 2 * (padding[0])+ 1) 
				+ image.at<uchar>(i, 2 *(padding[0]) - 1))
				+ coefficientsLP[2] * (image.at<uchar>(i, 2 *(padding[0]) + 2) 
				+ image.at<uchar>(i, 2 *(padding[0]) - 2))
				+ coefficientsLP[3] * (image.at<uchar>(i, 2 *(padding[0]) + 3) 
				+ image.at<uchar>(i, 2 *(padding[0]) - 3))
				+ coefficientsLP[4] * (ZERO_VALUE + image.at<uchar>(i, 2 *(padding[0]) - 4));

			new_image.at<uchar>(i,padding[1])  = coefficientsLP[0] * image.at<uchar>(i, 2 *(padding[1]))
				+ coefficientsLP[1] * (image.at<uchar>(i, 2 *(padding[1]) + 1) 
				+ image.at<uchar>(i, 2 *(padding[1]) - 1))
				+ coefficientsLP[2] * (ZERO_VALUE + image.at<uchar>(i, 2 *(padding[1]) - 2))
				+ coefficientsLP[3] * (ZERO_VALUE + image.at<uchar>(i, 2 *(padding[1]) - 3))
				+ coefficientsLP[4] * (ZERO_VALUE + image.at<uchar>(i, 2 * (padding[1]) - 4));

			new_image.at<uchar>(i,padding[2]) = ZERO_VALUE
				+ coefficientsLP[1] * (ZERO_VALUE + image.at<uchar>(i, 2 *(padding[2]) - 1))
				+ coefficientsLP[2] * (ZERO_VALUE + image.at<uchar>(i, 2 * (padding[2])- 2))
				+ coefficientsLP[3] * (ZERO_VALUE + image.at<uchar>(i, 2 * (padding[2]) - 3))
				+ coefficientsLP[4] * (ZERO_VALUE + image.at<uchar>(i, 2 * (padding[2])- 4));
				
				
				
					
			new_image.at<uchar>(i,padding[0]+offset) = coefficientsHP[0] * image.at<uchar>(i, 2 * padding[0])
				+ coefficientsHP[1] * (image.at<uchar>(i, 2 * (padding[0])+ 1) 
				+ image.at<uchar>(i, 2 *(padding[0]) - 1))
				+ coefficientsHP[2] * (image.at<uchar>(i, 2 *(padding[0]) + 2) 
				+ image.at<uchar>(i, 2 *(padding[0]) - 2))
				+ coefficientsHP[3] * (image.at<uchar>(i, 2 *(padding[0]) + 3) 
				+ image.at<uchar>(i, 2 *(padding[0]) - 3))
				+ coefficientsHP[4] * (ZERO_VALUE + image.at<uchar>(i, 2 *(padding[0]) - 4));

			new_image.at<uchar>(i,padding[1]+offset)  = coefficientsHP[0] * image.at<uchar>(i, 2 *(padding[1]))
				+ coefficientsHP[1] * (image.at<uchar>(i, 2 *(padding[1]) + 1) 
				+ image.at<uchar>(i, 2 *(padding[1]) - 1))
				+ coefficientsHP[2] * (ZERO_VALUE + image.at<uchar>(i, 2 *(padding[1]) - 2))
				+ coefficientsHP[3] * (ZERO_VALUE + image.at<uchar>(i, 2 *(padding[1]) - 3))
				+ coefficientsHP[4] * (ZERO_VALUE + image.at<uchar>(i, 2 * (padding[1]) - 4));

			new_image.at<uchar>(i,padding[2]+offset) = ZERO_VALUE
				+ coefficientsHP[1] * (ZERO_VALUE + image.at<uchar>(i, 2 *(padding[2]) - 1))
				+ coefficientsHP[2] * (ZERO_VALUE + image.at<uchar>(i, 2 * (padding[2]) - 2))
				+ coefficientsHP[3] * (ZERO_VALUE + image.at<uchar>(i, 2 * (padding[2]) - 3))
				+ coefficientsHP[4] * (ZERO_VALUE + image.at<uchar>(i, 2 * (padding[2]) - 4));
		}
		
		padding[0] = (imageHeight/ 2) - 2;
		padding[1] = (imageHeight/ 2) - 1;
		padding[2] = (imageHeight/ 2);
		
		offset = (imageHeight/2) - 1;
		
		#pragma omp parallel for private(i,j) schedule(static)
		for (i = 0; i < imageWidth ; i++) {
				
			tmp_image_1.at<uchar>(0,i) = coefficientsLP[0] * new_image.at<uchar>(0,i) 
				+ coefficientsLP[1] * (new_image.at<uchar>(1,i) + ZERO_VALUE)
				+ coefficientsLP[2] * (new_image.at<uchar>(2,i) + ZERO_VALUE)
				+ coefficientsLP[3] * (new_image.at<uchar>(3,i) + ZERO_VALUE)
				+ coefficientsLP[4] * (new_image.at<uchar>(4,i) + ZERO_VALUE);

			tmp_image_1.at<uchar>(1,i) = coefficientsLP[0] * new_image.at<uchar>(2,i) 
				+ coefficientsLP[1] * (new_image.at<uchar>(3,i) + new_image.at<uchar>(1,i))
				+ coefficientsLP[2] * (new_image.at<uchar>(4,i) + new_image.at<uchar>(0,i))
				+ coefficientsLP[3] * (new_image.at<uchar>(5,i) + ZERO_VALUE)
				+ coefficientsLP[4] * (new_image.at<uchar>(6,i) + ZERO_VALUE);

			tmp_image_1.at<uchar>(2,i) = coefficientsLP[0] * new_image.at<uchar>(4,i) 
				+ coefficientsLP[1] * (new_image.at<uchar>(5,i) + new_image.at<uchar>(3,i))
				+ coefficientsLP[2] * (new_image.at<uchar>(6,i) + new_image.at<uchar>(2,i))
				+ coefficientsLP[3] * (new_image.at<uchar>(7,i) + new_image.at<uchar>(1,i))
				+ coefficientsLP[4] * (new_image.at<uchar>(8,i) + new_image.at<uchar>(0,i));


			tmp_image_2.at<uchar>(0+offset,i) = coefficientsHP[0] * new_image.at<uchar>(0,i) 
				+ coefficientsHP[1] * (new_image.at<uchar>(1,i) + ZERO_VALUE)
				+ coefficientsHP[2] * (new_image.at<uchar>(2,i) + ZERO_VALUE)
				+ coefficientsHP[3] * (new_image.at<uchar>(3,i) + ZERO_VALUE)
				+ coefficientsHP[4] * (new_image.at<uchar>(4,i) + ZERO_VALUE);

			tmp_image_2.at<uchar>(1+offset,i) = coefficientsHP[0] * new_image.at<uchar>(2,i) 
				+ coefficientsHP[1] * (new_image.at<uchar>(3,i) + new_image.at<uchar>(1,i))
				+ coefficientsHP[2] * (new_image.at<uchar>(4,i) + new_image.at<uchar>(0,i))
				+ coefficientsHP[3] * (new_image.at<uchar>(5,i) + ZERO_VALUE)
				+ coefficientsHP[4] * (new_image.at<uchar>(6,i) + ZERO_VALUE);

			tmp_image_2.at<uchar>(2+offset,i) = coefficientsHP[0] * new_image.at<uchar>(4,i) 
				+ coefficientsHP[1] * (new_image.at<uchar>(5,i) + new_image.at<uchar>(3,i))
				+ coefficientsHP[2] * (new_image.at<uchar>(6,i) + new_image.at<uchar>(2,i))
				+ coefficientsHP[3] * (new_image.at<uchar>(7,i) + new_image.at<uchar>(1,i))
				+ coefficientsHP[4] * (new_image.at<uchar>(8,i) + new_image.at<uchar>(0,i));


			for (j = 3; j < (imageHeight/2) - 2; j++) {

				tmp_image_1.at<uchar>(j,i) = coefficientsLP[0] * new_image.at<uchar>(2*j,i)
					+ coefficientsLP[1] * (new_image.at<uchar>(2*j + 1,i) + new_image.at<uchar>(2*j - 1,i))
					+ coefficientsLP[2] * (new_image.at<uchar>(2*j + 2,i) + new_image.at<uchar>(2*j - 2,i))
					+ coefficientsLP[3] * (new_image.at<uchar>(2*j + 3,i) + new_image.at<uchar>(2*j - 3,i))
					+ coefficientsLP[4] * (new_image.at<uchar>(2*j + 4,i) + new_image.at<uchar>(2*j - 4,i));
					
					
				tmp_image_2.at<uchar>(j+offset,i) = coefficientsHP[0] * new_image.at<uchar>(2*j,i)
					+ coefficientsHP[1] * (new_image.at<uchar>(2*j + 1,i) + new_image.at<uchar>(2*j - 1,i))
					+ coefficientsHP[2] * (new_image.at<uchar>(2*j + 2,i) + new_image.at<uchar>(2*j - 2,i))
					+ coefficientsHP[3] * (new_image.at<uchar>(2*j + 3,i) + new_image.at<uchar>(2*j - 3,i))
					+ coefficientsHP[4] * (new_image.at<uchar>(2*j + 4,i) + new_image.at<uchar>(2*j - 4,i));
			}

			tmp_image_1.at<uchar>(padding[0],i) = coefficientsLP[0] * new_image.at<uchar>(2 * padding[0],i)
				+ coefficientsLP[1] * (new_image.at<uchar>(2 * (padding[0])+ 1,i) 
				+ new_image.at<uchar>(2 *(padding[0]) - 1,i))
				+ coefficientsLP[2] * (new_image.at<uchar>(2 *(padding[0]) + 2,i) 
				+ new_image.at<uchar>(2 *(padding[0]) - 2,i))
				+ coefficientsLP[3] * (new_image.at<uchar>(2 *(padding[0]) + 3,i) 
				+ new_image.at<uchar>(2 *(padding[0]) - 3,i))
				+ coefficientsLP[4] * (ZERO_VALUE + new_image.at<uchar>(2 *(padding[0]) - 4,i));

			tmp_image_1.at<uchar>(padding[1],i) = coefficientsLP[0] * new_image.at<uchar>(2*(padding[1]),i)
				+ coefficientsLP[1] * (new_image.at<uchar>(2 *(padding[1]) + 1,i) + new_image.at<uchar>(2 *(padding[1]) - 1,i))
				+ coefficientsLP[2] * (ZERO_VALUE + new_image.at<uchar>(2 *(padding[1]) - 2,i))
				+ coefficientsLP[3] * (ZERO_VALUE + new_image.at<uchar>(2 *(padding[1]) - 3,i))
				+ coefficientsLP[4] * (ZERO_VALUE + new_image.at<uchar>(2 * (padding[1]) - 4,i));

			tmp_image_1.at<uchar>(padding[2],i) = ZERO_VALUE
				+ coefficientsLP[1] * (ZERO_VALUE + new_image.at<uchar>(2 *(padding[2]) - 1,i))
				+ coefficientsLP[2] * (ZERO_VALUE + new_image.at<uchar>(2 * (padding[2])- 2,i))
				+ coefficientsLP[3] * (ZERO_VALUE + new_image.at<uchar>(2 * (padding[2]) - 3,i))
				+ coefficientsLP[4] * (ZERO_VALUE + new_image.at<uchar>(2 * (padding[2])- 4,i));
				
				
			tmp_image_2.at<uchar>(padding[0]+offset,i) = coefficientsHP[0] * new_image.at<uchar>(2 * padding[0],i)
				+ coefficientsHP[1] * (new_image.at<uchar>(2 * (padding[0])+ 1,i) 
				+ new_image.at<uchar>(2 *(padding[0]) - 1,i))
				+ coefficientsHP[2] * (new_image.at<uchar>(2 *(padding[0]) + 2,i) 
				+ new_image.at<uchar>(2 *(padding[0]) - 2,i))
				+ coefficientsHP[3] * (new_image.at<uchar>(2 *(padding[0]) + 3,i) 
				+ new_image.at<uchar>(2 *(padding[0]) - 3,i))
				+ coefficientsHP[4] * (ZERO_VALUE + new_image.at<uchar>(2 *(padding[0]) - 4,i));

			tmp_image_2.at<uchar>(padding[1]+offset,i) = coefficientsHP[0] * new_image.at<uchar>(2*(padding[1]),i)
				+ coefficientsHP[1] * (new_image.at<uchar>(2 *(padding[1]) + 1,i) + new_image.at<uchar>(2 *(padding[1]) - 1,i))
				+ coefficientsHP[2] * (ZERO_VALUE + new_image.at<uchar>(2 *(padding[1]) - 2,i))
				+ coefficientsHP[3] * (ZERO_VALUE + new_image.at<uchar>(2 *(padding[1]) - 3,i))
				+ coefficientsHP[4] * (ZERO_VALUE + new_image.at<uchar>(2 * (padding[1]) - 4,i));

			tmp_image_2.at<uchar>(padding[2]+offset,i) = ZERO_VALUE
				+ coefficientsHP[1] * (ZERO_VALUE + new_image.at<uchar>(2 *(padding[2]) - 1,i))
				+ coefficientsHP[2] * (ZERO_VALUE + new_image.at<uchar>(2 * (padding[2])- 2,i))
				+ coefficientsHP[3] * (ZERO_VALUE + new_image.at<uchar>(2 * (padding[2]) - 3,i))
				+ coefficientsHP[4] * (ZERO_VALUE + new_image.at<uchar>(2 * (padding[2])- 4,i));
			}
			
			
	final_time = omp_get_wtime();
	final_time -= initial_time;	
	printf("time %lf\n", final_time);

//    Copia di tmp_image_1 e tmp_image_2 in new_image
    for(int i=0;i<imageWidth;i++){
        for(int j=0;j<imageHeight/2;j++){
            new_image.at<uchar>(j,i) = tmp_image_1.at<uchar>(j,i);
        }
        for(int j=imageHeight/2;j<imageHeight;j++){
            new_image.at<uchar>(j,i) = tmp_image_2.at<uchar>(j,i);
        }
    }
    
    	cv::resize(new_image,new_image, cv::Size(1920,1080),0,0,cv::INTER_LINEAR);

	imshow("CDF parallela",new_image);
	
//	vector<int> compression_params;

//	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);

//	compression_params.push_back(60);

//	imwrite("./immagini_modificate/CDF_9_7_parallela/CDF_9_7_parallela.jpg",new_image,compression_params);

	waitKey(0);

	return 0;
}


uchar wavelet_row(Mat src,int i, int j,double coefficients[]){
	
	uchar out=0;
	
	out = coefficients[0] * src.at<uchar>(i, 2*j)
		+ coefficients[1] * (src.at<uchar>(i, 2*j + 1) + src.at<uchar>(i, 2*j - 1))
		+ coefficients[2] * (src.at<uchar>(i, 2*j + 2) + src.at<uchar>(i, 2*j - 2))
		+ coefficients[3] * (src.at<uchar>(i, 2*j + 3) + src.at<uchar>(i, 2*j - 3))
		+ coefficients[4] * (src.at<uchar>(i, 2*j + 4) + src.at<uchar>(i, 2*j - 4));
		
	return out;
	
}





//void wavelet_row(Mat src, Mat dst, double coefficients[],int offset){
//	
//	int srcWidth = src.size().width;
//	int srcHeight = src.size().height;
//	
//	int padding[3] = {(srcWidth/2) - 2 ,(srcWidth/2) - 1 ,(srcWidth/2)};
//	
//	int i=0,j=0;

//	#pragma omp parallel for private(i,j) schedule(static)
//	for (i = 0; i < srcHeight ; i++) {
//				
//		printf("Thread ID:%i\n",omp_get_thread_num());
//					
//		dst.at<uchar>(i,0 + offset) = coefficients[0] * src.at<uchar>(i,0) 
//			+ coefficients[1] * (src.at<uchar>(i, 1) + ZERO_VALUE)
//			+ coefficients[2] * (src.at<uchar>(i, 2) + ZERO_VALUE)
//			+ coefficients[3] * (src.at<uchar>(i, 3) + ZERO_VALUE)
//			+ coefficients[4] * (src.at<uchar>(i, 4) + ZERO_VALUE);

//		dst.at<uchar>(i,1 + offset) = coefficients[0] * src.at<uchar>(i, 2) 
//			+ coefficients[1] * (src.at<uchar>(i, 3) + src.at<uchar>(i, 1))
//			+ coefficients[2] * (src.at<uchar>(i, 4) + src.at<uchar>(i, 0))
//			+ coefficients[3] * (src.at<uchar>(i, 5) + ZERO_VALUE)
//			+ coefficients[4] * (src.at<uchar>(i, 6) + ZERO_VALUE);

//		dst.at<uchar>(i,2 + offset) = coefficients[0] * src.at<uchar>(i, 4) 
//			+ coefficients[1] * (src.at<uchar>(i, 5) + src.at<uchar>(i, 3))
//			+ coefficients[2] * (src.at<uchar>(i, 6) + src.at<uchar>(i, 2))
//			+ coefficients[3] * (src.at<uchar>(i, 7) + src.at<uchar>(i, 1))
//			+ coefficients[4] * (src.at<uchar>(i, 8) + src.at<uchar>(i, 0));


//		for (j = 3; j < (srcWidth/2) - 2; j++) {

//			dst.at<uchar>(i,j + offset) = coefficients[0] * src.at<uchar>(i, 2*j)
//				+ coefficients[1] * (src.at<uchar>(i, 2*j + 1) + src.at<uchar>(i, 2*j - 1))
//				+ coefficients[2] * (src.at<uchar>(i, 2*j + 2) + src.at<uchar>(i, 2*j - 2))
//				+ coefficients[3] * (src.at<uchar>(i, 2*j + 3) + src.at<uchar>(i, 2*j - 3))
//				+ coefficients[4] * (src.at<uchar>(i, 2*j + 4) + src.at<uchar>(i, 2*j - 4));
//		}

//		dst.at<uchar>(i,padding[0]+offset) = coefficients[0] * src.at<uchar>(i, 2 * padding[0])
//			+ coefficients[1] * (src.at<uchar>(i, 2 * (padding[0])+ 1) 
//			+ src.at<uchar>(i, 2 *(padding[0]) - 1))
//			+ coefficients[2] * (src.at<uchar>(i, 2 *(padding[0]) + 2) 
//			+ src.at<uchar>(i, 2 *(padding[0]) - 2))
//			+ coefficients[3] * (src.at<uchar>(i, 2 *(padding[0]) + 3) 
//			+ src.at<uchar>(i, 2 *(padding[0]) - 3))
//			+ coefficients[4] * (ZERO_VALUE + src.at<uchar>(i, 2 *(padding[0]) - 4));

//		dst.at<uchar>(i,padding[1]+offset)  = coefficients[0] * src.at<uchar>(i, 2 *(padding[1]))
//			+ coefficients[1] * (src.at<uchar>(i, 2 *(padding[1]) + 1) 
//			+ src.at<uchar>(i, 2 *(padding[1]) - 1))
//			+ coefficients[2] * (ZERO_VALUE + src.at<uchar>(i, 2 *(padding[1]) - 2))
//			+ coefficients[3] * (ZERO_VALUE + src.at<uchar>(i, 2 *(padding[1]) - 3))
//			+ coefficients[4] * (ZERO_VALUE + src.at<uchar>(i, 2 * (padding[1]) - 4));

//		dst.at<uchar>(i,padding[2]+offset) = ZERO_VALUE
//			+ coefficients[1] * (ZERO_VALUE + src.at<uchar>(i, 2 *(padding[2]) - 1))
//			+ coefficients[2] * (ZERO_VALUE + src.at<uchar>(i, 2 * (padding[2])- 2))
//			+ coefficients[3] * (ZERO_VALUE + src.at<uchar>(i, 2 * (padding[2]) - 3))
//			+ coefficients[4] * (ZERO_VALUE + src.at<uchar>(i, 2 * (padding[2])- 4));
//	}
//}


//void wavelet_cols(Mat src,Mat dst, double coefficients[],int offset){

//	int srcWidth = src.size().width;
//	int srcHeight = src.size().height;
//	
//	int padding[3] = {(srcHeight/ 2) - 2 ,(srcHeight/ 2) - 1 ,(srcHeight/ 2)};
//	
//	int i=0,j=0;

//	#pragma omp parallel for private(i,j) schedule(static)
//	for (i = 0; i < srcWidth ; i++) {
//			
//		dst.at<uchar>(0+offset,i) = coefficients[0] * src.at<uchar>(0,i) 
//			+ coefficients[1] * (src.at<uchar>(1,i) + ZERO_VALUE)
//			+ coefficients[2] * (src.at<uchar>(2,i) + ZERO_VALUE)
//			+ coefficients[3] * (src.at<uchar>(3,i) + ZERO_VALUE)
//			+ coefficients[4] * (src.at<uchar>(4,i) + ZERO_VALUE);

//		dst.at<uchar>(1+offset,i) = coefficients[0] * src.at<uchar>(2,i) 
//			+ coefficients[1] * (src.at<uchar>(3,i) + src.at<uchar>(1,i))
//			+ coefficients[2] * (src.at<uchar>(4,i) + src.at<uchar>(0,i))
//			+ coefficients[3] * (src.at<uchar>(5,i) + ZERO_VALUE)
//			+ coefficients[4] * (src.at<uchar>(6,i) + ZERO_VALUE);

//		dst.at<uchar>(2+offset,i) = coefficients[0] * src.at<uchar>(4,i) 
//			+ coefficients[1] * (src.at<uchar>(5,i) + src.at<uchar>(3,i))
//			+ coefficients[2] * (src.at<uchar>(6,i) + src.at<uchar>(2,i))
//			+ coefficients[3] * (src.at<uchar>(7,i) + src.at<uchar>(1,i))
//			+ coefficients[4] * (src.at<uchar>(8,i) + src.at<uchar>(0,i));


//		for (j = 3; j < (srcHeight/2) - 2; j++) {

//			dst.at<uchar>(j+offset,i) = coefficients[0] * src.at<uchar>(2*j,i)
//				+ coefficients[1] * (src.at<uchar>(2*j + 1,i) + src.at<uchar>(2*j - 1,i))
//				+ coefficients[2] * (src.at<uchar>(2*j + 2,i) + src.at<uchar>(2*j - 2,i))
//				+ coefficients[3] * (src.at<uchar>(2*j + 3,i) + src.at<uchar>(2*j - 3,i))
//				+ coefficients[4] * (src.at<uchar>(2*j + 4,i) + src.at<uchar>(2*j - 4,i));
//		}

//		dst.at<uchar>(padding[0]+offset,i) = coefficients[0] * src.at<uchar>(2 * padding[0],i)
//			+ coefficients[1] * (src.at<uchar>(2 * (padding[0])+ 1,i) 
//			+ src.at<uchar>(2 *(padding[0]) - 1,i))
//			+ coefficients[2] * (src.at<uchar>(2 *(padding[0]) + 2,i) 
//			+ src.at<uchar>(2 *(padding[0]) - 2,i))
//			+ coefficients[3] * (src.at<uchar>(2 *(padding[0]) + 3,i) 
//			+ src.at<uchar>(2 *(padding[0]) - 3,i))
//			+ coefficients[4] * (ZERO_VALUE + src.at<uchar>(2 *(padding[0]) - 4,i));

//		dst.at<uchar>(padding[1]+offset,i) = coefficients[0] * src.at<uchar>(2*(padding[1]),i)
//			+ coefficients[1] * (src.at<uchar>(2 *(padding[1]) + 1,i) + src.at<uchar>(2 *(padding[1]) - 1,i))
//			+ coefficients[2] * (ZERO_VALUE + src.at<uchar>(2 *(padding[1]) - 2,i))
//			+ coefficients[3] * (ZERO_VALUE + src.at<uchar>(2 *(padding[1]) - 3,i))
//			+ coefficients[4] * (ZERO_VALUE + src.at<uchar>(2 * (padding[1]) - 4,i));

//	/*	dst.at<uchar>(padding[2]+offset,i) = ZERO_VALUE
//			+ coefficients[1] * (ZERO_VALUE + src.at<uchar>(2 *(padding[2]) - 1,i))
//			+ coefficients[2] * (ZERO_VALUE + src.at<uchar>(2 * (padding[2])- 2,i))
//			+ coefficients[3] * (ZERO_VALUE + src.at<uchar>(2 * (padding[2]) - 3,i))
//			+ coefficients[4] * (ZERO_VALUE + src.at<uchar>(2 * (padding[2])- 4,i));*/
//		}
//}

























