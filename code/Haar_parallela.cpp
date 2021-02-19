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
________________________________________________Haar Transform_________________________________________________________
*/

Mat Haar_antitrasformata(Mat immagine_trasformata);


int main(int argc, char* argv[]) {
	const char* imgName = argv[1];
	Mat image = imread(imgName, IMREAD_GRAYSCALE);
	image.convertTo(image,CV_64FC1);
	
	Mat dst = cv::Mat(image.rows,image.cols,CV_64FC1);
	int imageWidth = image.size().width;
	int w = 0;
	int k = 0;
	
	double initial_time = 0, final_time = 0;
	int temp_row = 0;
	int temp_col = 0;

	int i=0;
	int j=0;
	int t=0;
	int nOfLevels = atoi(argv[2]);
	
	temp_row = (image.rows);
	temp_col = (image.cols);
	
	w = (image.cols);
	k = (image.rows);
	
	double time_A=0,time_D=0,time_AA=0,time_DD=0;
	
	initial_time = omp_get_wtime();
	
	//immagine A e D			
	omp_set_num_threads(omp_get_max_threads());
	time_A = omp_get_wtime();
	for(t=0;t<nOfLevels;t++){
	
	#pragma omp parallel for private(i,j) schedule(static)
	for (i = 0; i < k; i++) 
	{
		for (j = 0; j < w/2; j++) {
			dst.at<double>(i, j) = (image.at<double>(i,2*j) + image.at<double>(i,2*j + 1)) / 2;
			
			dst.at<double>(i, j+(w/2)) = (image.at<double>(i,2*j) - image.at<double>(i,2*j+1))/2;
		}
	}
		
	time_D = omp_get_wtime();
	time_D -= time_A;

	time_AA = omp_get_wtime();
	#pragma omp parallel for private(i,j) schedule(static)
	for (i = 0; i < w; i++) {		
		for (j = 0; j < k/2; j++) {
			image.at<double>(j, i) = (dst.at<double>(2*j,i) + dst.at<double>(2*j + 1, i)) / 2;
			image.at<double>(j+(k/2), i) = (dst.at<double>(2*j,i) - dst.at<double>(2*j+1,i))/2;
		}
	}
	
		k = k/2;
		w = w/2;	

	}

	time_DD = omp_get_wtime();
	time_DD -= time_AA;
	
	final_time = omp_get_wtime();
	final_time -= initial_time;

	printf("time: %lf \n", final_time);
	printf("time A e D: %lf \n", time_D);
	printf("time AA e DD: %lf \n", time_DD);

//Instructions to save the final image on disk 
/*	vector<int> compression_params;

	compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);

	compression_params.push_back(60);

	imwrite("./immagini_modificate/Haar_parallela/Haar_parallela.jpg",image,compression_params);
	imwrite("./immagini_modificate/Haar_parallela/Haar_parallela_antitrasformata.jpg",Haar_antitrasformata(image),compression_params);
	
	waitKey(0);*/
	
	
	cv::Mat final_image(image.rows, image.cols, CV_8UC1);
	
	
	cv::normalize(image, final_image, 0, 255, NORM_MINMAX,CV_8UC1);
	
	imshow("Parallel Haar",final_image);
	waitKey(0);
	return 0;
}

//Haar antitransform
Mat Haar_antitrasformata(Mat immagine_trasformata){
	
	Mat image_out = cv::Mat(immagine_trasformata.rows, immagine_trasformata.cols, CV_8UC1);
	
	Mat image_out_2 = cv::Mat(immagine_trasformata.rows, immagine_trasformata.cols, CV_8UC1);

	
	int k = immagine_trasformata.rows;
	int w = immagine_trasformata.cols;
 	int i,j;	

	#pragma omp parallel for private(i,j) schedule(static)
	for(i=0;i<k/2;i++){
		for(j=0;j<w;j++){
			image_out.at<double>(2*i,j) = immagine_trasformata.at<double>(i,j) + immagine_trasformata.at<double>(i+(k/2),j);
			
			image_out.at<double>(2*i+1,j) = immagine_trasformata.at<double>(i,j) -  immagine_trasformata.at<double>(i+(k/2),j);
			
		}
	}
	
	#pragma omp parallel for private(i,j) schedule(static)
	for(i=0;i<k;i++){
		for(j=0;j<w/2;j++){
			image_out_2.at<double>(i,2*j) = image_out.at<double>(i,j) + image_out.at<double>(i,j+(w/2));
			
			image_out_2.at<double>(i,2*j+1) = image_out.at<double>(i,j) -  image_out.at<double>(i,j+(w/2));
			
		}
	}
	return image_out_2;
	
}





