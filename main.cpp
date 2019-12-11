#include <vector>
#include <queue>
#include <math.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/photo.hpp>

using namespace std;
using namespace cv;

int main()  // int argc, char *argv[]
{
    // --- Read image
    Mat img = imread( "../Lab3_Harris/image_harris.jpg" );
    imshow( "image", img );
    
    
    // --- --- Opencv implementation
    // --- Convert to gtay
    Mat img_gray;
    cvtColor( img, img_gray, COLOR_BGR2GRAY );
    
    // --- Find corners
    Mat corners_cv = Mat::zeros( img_gray.size(), CV_32FC1 );
    cornerHarris( img_gray, corners_cv, 3, 3, 0.05, BORDER_DEFAULT );
    Mat corners_cv_norm, dst_norm_scaled;
    normalize( corners_cv, corners_cv_norm, 0, 255, NORM_MINMAX, CV_32FC1 );
    convertScaleAbs( corners_cv_norm, dst_norm_scaled );
    imshow( "opnecv corners", dst_norm_scaled );
    imwrite( "../Lab3_Harris/dst_norm_scaled.png", dst_norm_scaled );
    
    // --- Drawing a circle around corners
    Mat img_corners = Mat( img.size(), img.type(), img.data );
    int thresh = 140;       // Threshold for corners
    for( int j = 0; j < corners_cv_norm.rows ; j++ )
        for( int i = 0; i < corners_cv_norm.cols; i++ )
            if(  corners_cv_norm.at<float>(j,i) > thresh )
               circle( img_corners, Point( i, j ), 4,  Scalar(255, 0, 0), 1, LINE_4, 0 );
    imshow( "corners on image", img_corners );
    imwrite( "../Lab3_Harris/img_corners.png", img_corners );
    
    // --- --- Our Harris Detector Implementation
    
    
    
    waitKey(0);
    return 0;
}
