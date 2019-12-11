//#include <iostream>
//#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

void SobelEdge( Mat imgin, Mat &imgSobelXX, Mat &imgSobelYY, Mat &imgSobelXY )
{
	imgin.convertTo( imgin, CV_32F );
	imgSobelXX = Mat::zeros( imgin.size(), CV_32F );
	imgSobelYY = Mat::zeros( imgin.size(), CV_32F );
	imgSobelXY = Mat::zeros( imgin.size(), CV_32F );
    float Sx[3][3] = { { -1,-2,-1 },
                       {  0, 0, 0 },
                       {  1, 2, 1 } };
    float Sy[3][3] = { { -1, 0, 1 },
                       { -2, 0, 2 },
                       { -1, 0, 1 } };
    
	for (int i = 1; i < imgin.cols - 1; ++i) 
    {
		for (int j = 1; j < imgin.rows - 1; ++j) 
        {
            float SumX = 0, SumY = 0;
            for (int ii = -1; ii <= 1; ii++) 
            {
                for (int jj = -1; jj <= 1; jj++)
                {
                    SumX += Sx[jj + 1][ii + 1] * imgin.at<float>(j + jj, i + ii);
                    SumY += Sy[jj + 1][ii + 1] * imgin.at<float>(j + jj, i + ii);
                }
            }
            SumX = abs(SumX);
            SumY = abs(SumY);
            
            imgSobelXX.at<float>(j, i) = SumX * SumX;
			imgSobelYY.at<float>(j, i) = SumY * SumY;
			imgSobelXY.at<float>(j, i) = SumX * SumY;
		}
	}
}

void HandMadeHarris( Mat &imgSobelXX, Mat &imgSobelYY, Mat &imgSobelXY, Mat &imgres, float k )
{
	imgres = Mat::zeros( imgSobelXX.size(), CV_32F );
	for (int i = 1; i < imgSobelXX.cols - 1; ++i)
    {
		for (int j = 1; j < imgSobelXX.rows - 1; ++j)
        {
			float a = imgSobelXX.at<float>(j, i);
			float b = imgSobelYY.at<float>(j, i);
			float c = imgSobelXY.at<float>(j, i);			
			imgres.at<float>(j, i) = a * b - c * c - k * (a + b) * (a + b);
		}
	}
}

void drawCorners( Mat src, Mat &dst, Mat corn, int thresh, Scalar color )
{
    src.copyTo( dst );
    for( int j = 0; j < corn.rows ; j++ )
        for( int i = 0; i < corn.cols; i++ )
            if(  corn.at<float>(j,i) > thresh )
               circle( dst, Point( i, j ), 4,  color, 1, LINE_4, 0 );
}

int main()  // int argc, char *argv[]
{
    // --- Read image
    Mat img = imread( "../Lab3_Harris/image_harris.jpg" );
    imshow( "image", img );
    
    // --- Convert to gtay
    Mat img_gray;
    cvtColor( img, img_gray, COLOR_BGR2GRAY );
    
    // --- --- Opencv implementation
    // --- Find corners
    float k = 0.05f;
    Mat corners_cv = Mat::zeros( img_gray.size(), CV_32FC1 );
    cornerHarris( img_gray, corners_cv, 3, 3, double(k), BORDER_DEFAULT );
    
    // --- normalize and save
    Mat corners_norm_cv;
    normalize( corners_cv, corners_norm_cv, 0, 255, NORM_MINMAX, CV_32FC1 );
    Mat dst_norm_scaled_cv;
    convertScaleAbs( corners_norm_cv, dst_norm_scaled_cv );
    imshow( "opnecv corners", dst_norm_scaled_cv );
    imwrite( "../Lab3_Harris/dst_norm_scaled_cv.png", dst_norm_scaled_cv );
    
    // --- Drawing a circle around corners and save
    Mat img_corners_cv;
    int thresh = 140;       // Threshold for corners
    drawCorners( img, img_corners_cv, corners_norm_cv, thresh, Scalar(255,0,0) );
    imshow( "corners on image by opencv", img_corners_cv );
    imwrite( "../Lab3_Harris/img_corners_cv.png", img_corners_cv );
    
    
    // --- --- Our Harris Detector Implementation
    // --- Find corners
    Mat imgSobelXX, imgSobelYY, imgSobelXY, corners_my;
    SobelEdge( img_gray, imgSobelXX, imgSobelYY, imgSobelXY );
    GaussianBlur( imgSobelXX, imgSobelXX, Size(5,5), 1.0, 1.0 );
    GaussianBlur( imgSobelYY, imgSobelYY, Size(5, 5), 1.0, 1.0 );
    GaussianBlur( imgSobelXY, imgSobelXY, Size(5, 5), 1.0, 1.0 );
    HandMadeHarris( imgSobelXX, imgSobelYY, imgSobelXY, corners_my, k );
    
    // --- normalize and save
    Mat corners_norm_my;
    normalize( corners_my, corners_norm_my, 0, 255, NORM_MINMAX );
    Mat dst_norm_scaled_my;
    convertScaleAbs( corners_norm_my, dst_norm_scaled_my );
    imshow( "my corners", dst_norm_scaled_my );
    imwrite( "../Lab3_Harris/dst_norm_scaled_my.png", dst_norm_scaled_my );
    
    // --- Drawing a circle around corners and save
    Mat img_corners_my;
    int thresh2 = 160;
    drawCorners( img, img_corners_my, corners_norm_my, thresh2, Scalar(0,0,255) );
    imshow( "corners on image by me", img_corners_my );
    imwrite( "../Lab3_Harris/img_corners_my.png", img_corners_my );
    
    
    waitKey(0);
    return 0;
}
