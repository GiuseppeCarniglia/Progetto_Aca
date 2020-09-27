

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


int main(int argv, char *argc[])
{
	const char* imgName = argc[1];
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
	
		

	for (i = 0; i < image.rows ; i++) {
		
		//Calcolo matrice di approssimazione a
		a[0] = coefficientsLP[0] * image.at<uchar>(i,0) 
			+ coefficientsLP[1] * (image.at<uchar>(i, 1) + ZERO_VALUE)
			+ coefficientsLP[2] * (image.at<uchar>(i, 2) + ZERO_VALUE)
			+ coefficientsLP[3] * (image.at<uchar>(i, 3) + ZERO_VALUE)
			+ coefficientsLP[4] * (image.at<uchar>(i, 4) + ZERO_VALUE);

		a[1] = coefficientsLP[0] * image.at<uchar>(i, 2) 
			+ coefficientsLP[1] * (image.at<uchar>(i, 3) + image.at<uchar>(i, 1))
			+ coefficientsLP[2] * (image.at<uchar>(i, 4) + image.at<uchar>(i, 0))
			+ coefficientsLP[3] * (image.at<uchar>(i, 5) + ZERO_VALUE)
			+ coefficientsLP[4] * (image.at<uchar>(i, 6) + ZERO_VALUE);

		a[2] = coefficientsLP[0] * image.at<uchar>(i, 4) 
			+ coefficientsLP[1] * (image.at<uchar>(i, 5) + image.at<uchar>(i, 3))
			+ coefficientsLP[2] * (image.at<uchar>(i, 6) + image.at<uchar>(i, 2))
			+ coefficientsLP[3] * (image.at<uchar>(i, 7) + image.at<uchar>(i, 1))
			+ coefficientsLP[4] * (image.at<uchar>(i, 8) + image.at<uchar>(i, 0));


			for (j = 3; j < (w/2) - 2; j++) {

				a[j] = coefficientsLP[0] * image.at<uchar>(i, 2*j)
					+ coefficientsLP[1] * (image.at<uchar>(i, 2*j + 1) + image.at<uchar>(i, 2*j - 1))
					+ coefficientsLP[2] * (image.at<uchar>(i, 2*j + 2) + image.at<uchar>(i, 2*j - 2))
					+ coefficientsLP[3] * (image.at<uchar>(i, 2*j + 3) + image.at<uchar>(i, 2*j - 3))
					+ coefficientsLP[4] * (image.at<uchar>(i, 2*j + 4) + image.at<uchar>(i, 2*j - 4));
			}

		a[padding[0]] = coefficientsLP[0] * image.at<uchar>(i, 2 * padding[0])
			+ coefficientsLP[1] * (image.at<uchar>(i, 2 * padding[0] + 1) 
			+ image.at<uchar>(i, 2 * padding[0] - 1))
			+ coefficientsLP[2] * (image.at<uchar>(i, 2 * padding[0] + 2) 
			+ image.at<uchar>(i, 2 * padding[0] - 2))
			+ coefficientsLP[3] * (image.at<uchar>(i, 2 * padding[0] + 3) 
			+ image.at<uchar>(i, 2 * padding[0] - 3))
			+ coefficientsLP[4] * (ZERO_VALUE + image.at<uchar>(i, 2 * padding[0] - 4));

		a[padding[1]] = coefficientsLP[0] * image.at<uchar>(i, 2 * padding[1])
			+ coefficientsLP[1] * (image.at<uchar>(i, 2 * padding[1] + 1) 
			+ image.at<uchar>(i, 2 * padding[1] - 1))
			+ coefficientsLP[2] * (ZERO_VALUE + image.at<uchar>(i, 2 * padding[1] - 2))
			+ coefficientsLP[3] * (ZERO_VALUE + image.at<uchar>(i, 2 * padding[1] - 3))
			+ coefficientsLP[4] * (ZERO_VALUE + image.at<uchar>(i, 2 * padding[1] - 4));

		a[padding[2]] = ZERO_VALUE
			+ coefficientsLP[1] * (ZERO_VALUE + image.at<uchar>(i, 2 * padding[2] - 1))
			+ coefficientsLP[2] * (ZERO_VALUE + image.at<uchar>(i, 2 * padding[2] - 2))
			+ coefficientsLP[3] * (ZERO_VALUE + image.at<uchar>(i, 2 * padding[2] - 3))
			+ coefficientsLP[4] * (ZERO_VALUE + image.at<uchar>(i, 2 * padding[2] - 4));
		
		
		for(j = 0; j < (imageWidth/2) ; j++) {
				new_image.at<uchar>(i, j) = a[j];
			}
		}

	for (i = 0; i < image.rows ; i++) {
		//Calcolo matrice di dettaglio d

		 d[0] = coefficientsHP[0] * image.at<uchar>(i, 0)
			 + coefficientsHP[1] * (image.at<uchar>(i, 1) + ZERO_VALUE)
			 + coefficientsHP[2] * (image.at<uchar>(i, 2) + ZERO_VALUE)
			 + coefficientsHP[3] * (image.at<uchar>(i, 3) + ZERO_VALUE)
			 + coefficientsHP[4] * (image.at<uchar>(i, 4) + ZERO_VALUE);

		 d[1] = coefficientsHP[0] * image.at<uchar>(i, 2)
			 + coefficientsHP[1] * (image.at<uchar>(i, 3) + image.at<uchar>(i, 1))
			 + coefficientsHP[2] * (image.at<uchar>(i, 4) + image.at<uchar>(i, 0))
			 + coefficientsHP[3] * (image.at<uchar>(i, 5) + ZERO_VALUE)
			 + coefficientsHP[4] * (image.at<uchar>(i, 6) + ZERO_VALUE);

		d[2] = coefficientsHP[0] * image.at<uchar>(i, 4)
			+ coefficientsHP[1] * (image.at<uchar>(i, 5) + image.at<uchar>(i, 3))
			+ coefficientsHP[2] * (image.at<uchar>(i, 6) + image.at<uchar>(i, 2))
			+ coefficientsHP[3] * (image.at<uchar>(i, 7) + image.at<uchar>(i, 1))
			+ coefficientsHP[4] * (image.at<uchar>(i, 8) + image.at<uchar>(i, 0));

		
		for (j = 3; j < (imageWidth/2) - 2; j++) {
			 d[j] = coefficientsHP[0] * image.at<uchar>(i, 2*j)
				+ coefficientsHP[1] * (image.at<uchar>(i, 2*j + 1)+image.at<uchar>(i, 2*j - 1))
				+ coefficientsHP[2] * (image.at<uchar>(i, 2*j + 2)+image.at<uchar>(i, 2*j - 2))
				+ coefficientsHP[3] * (image.at<uchar>(i, 2*j + 3)+image.at<uchar>(i, 2*j - 3))
				+ coefficientsHP[4] * (image.at<uchar>(i, 2*j + 4)+image.at<uchar>(i, 2*j - 4));
		}
		
		d[padding[0]] = coefficientsHP[0] * image.at<uchar>(i, 2*padding[0])
			+ coefficientsHP[1] * (image.at<uchar>(i, 2*padding[0]+1) 
				+ image.at<uchar>(i, 2*padding[0]-1))
			+ coefficientsHP[2] * (image.at<uchar>(i, 2*padding[0]+2) 
				+ image.at<uchar>(i, 2*padding[0]-2))
			+ coefficientsHP[3] * (image.at<uchar>(i, 2*padding[0]+3) 
				+ image.at<uchar>(i, 2*padding[0]-3))
			+ coefficientsHP[4] * (ZERO_VALUE + image.at<uchar>(i, 2*padding[0]-4));
		
		d[padding[1]] = coefficientsHP[0] * image.at<uchar>(i, 2*padding[1])
			+ coefficientsHP[1] * (image.at<uchar>(i, 2*padding[1]+1)
				+ image.at<uchar>(i, 2*padding[1]-1))
			+ coefficientsHP[2] * (image.at<uchar>(i, 2*padding[1]+2)
				+ image.at<uchar>(i, 2*padding[1]-2))
			+ coefficientsHP[3] * (ZERO_VALUE + image.at<uchar>(i, 2*padding[1]-3))
			+ coefficientsHP[4] * (ZERO_VALUE + image.at<uchar>(i, 2*padding[1]-4));
		
		d[padding[2]] = coefficientsHP[0] * image.at<uchar>(i, 2*padding[2])
			+ coefficientsHP[1] * (image.at<uchar>(i, 2*padding[2]+1)+image.at<uchar>(i, 2*padding[2]-1))
			+ coefficientsHP[2] * (ZERO_VALUE + image.at<uchar>(i, 2*padding[2]-2))
			+ coefficientsHP[3] * (ZERO_VALUE + image.at<uchar>(i, 2*padding[2]-3))
			+ coefficientsHP[4] * (ZERO_VALUE + image.at<uchar>(i, 2*padding[2]-4));
		
		for(k = 0; k<(imageWidth/2); k++) {
			new_image.at<uchar>(i,k+(imageWidth/2)) = d[k];
		}
	}

			
			
	free(a);
	free(d);
	final_time = omp_get_wtime();
	final_time -= initial_time;
	
	printf("time %lf\n", final_time);

	vector<int> compression_params;

	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);

	compression_params.push_back(60);

	imwrite("./immagini_modificate/CDF_9_7_seriale/CDF_9_7_seriale.jpg",new_image,compression_params);
	waitKey(0);

	return 0;
}




