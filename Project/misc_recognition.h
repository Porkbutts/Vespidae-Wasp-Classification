#ifndef PRIME_PROJECT_2011_MISC_RECOGNITION_H
#define PRIME_PROJECT_2011_MISC_RECOGNITION_H
#include <cv.h>
#include <ml.h>

// Image Processing
void binarize( const cv::Mat& src, cv::Mat& dst );

cv::Rect getBoundingRect( const cv::Mat& bin );


// Color Feature extraction
cv::Mat calcColorWindow( const cv::Mat& src, int blockLength, int blockStride );

void calcColorFeature( const cv::Mat& src, cv::Mat& dst, const cv::Mat& mask = cv::Mat() );


// Machine learning
int getNumLabels( const cv::Mat& labels );

cv::Mat mergeAllExcept( const std::vector<cv::Mat>& indexSets, int fold );

std::vector<cv::Mat> splitData( const cv::Mat& labels, int k );

std::vector<cv::SVM> oneVsAllSVM_train( const cv::Mat& features, const cv::Mat& labels, 
	const cv::Mat& varIdx, const cv::Mat& sampleIdx, cv::SVMParams params );

float oneVsAllSVM_predict( const std::vector<cv::SVM>& svms, 
	const cv::Mat& feature, cv::Mat& confidenceVector );

float predictFromJointProb( const cv::Mat& colorProbs, 
	const cv::Mat& hogProbs, bool returnConfidence = false );

float calculateAcc( const cv::Mat& confusionMatrix,
	float* correct = 0, float* total = 0, int cl = -1 );

void printResults( const cv::Mat& confusionMatrix );


#endif