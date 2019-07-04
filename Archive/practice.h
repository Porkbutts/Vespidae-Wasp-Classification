#ifndef PRIME_PROJECT_2011_PRACTICE_H
#define PRIME_PROJECT_2011_PRACTICE_H
#include <cv.h>
#include <ml.h>

/** These functions are mainly for demonstration and learning purposes only.
  * I struggled to understand how many of the OpenCV classes were used,
  * such as SVM and HOGDescriptor. After doing my own implementations, I had
  * better understanding of the concepts and was able to set the right
  * parameters and such for the OpenCV implementations.
  **/

// Moments analysis
cv::Point findCentroid( const cv::Mat& src );

// Support Vector Machine
std::vector<std::vector<cv::SVM>> oneVsOneSVM_train( const cv::Mat& trainFeatures, 
	const cv::Mat& trainLabels, int numLabels, cv::SVMParams params );

float oneVsOneSVM_predict( const std::vector<std::vector<cv::SVM>>& svms, const cv::Mat& feature );

std::vector<cv::SVM> oneVsAllSVM_train( const cv::Mat& trainFeatures,
	const cv::Mat& trainLabels, int numLabels, cv::SVMParams params );

float oneVsAllSVM_predict( const std::vector<cv::SVM>& svms, const cv::Mat& feature );

// Histogram of Oriented Gradients
cv::Mat drawHog( const cv::Mat& src, int len = 16 );

std::vector<cv::Mat> calculateIntegralHOG( const cv::Mat& src );

void calculateHOG_rect( cv::Rect cell, cv::Mat& hog_cell, 
	const std::vector<cv::Mat>& integrals, int normalization );

void calculateHOG_block( cv::Rect block, cv::Mat& hog_block, 
	const std::vector<cv::Mat>& integrals, cv::Size cell, int normalization );

cv::Mat calculateHOG_window( const std::vector<cv::Mat>& integrals, 
	cv::Rect window, int normalization = 4 );


#endif