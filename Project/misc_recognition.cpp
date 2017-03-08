/******************************************************************************
 * File:			misc_recognition.cpp
 * Author:			Adrian Teng-Amnuay
 * Email:			atengamn@ucsd.edu, pumpadrian@gmail.com
 *
 * Description:		This program is open source, and was created to assist
 *					in automated Vespidae wasp recognition for TFRI, Taiwan.
 *
 *					The program contains functions used by the wasp
 *					classifier for image pre-processing, color feature
 *					extraction, and model validation methods.
 *****************************************************************************/
#include "misc_recognition.h"
#include <iostream>
using namespace cv;
using namespace std;

// Color feature parameters
#define hbins 30
#define sbins 32
static const int colorFeatureLength = (hbins * sbins);

/**
 * binarize
 *
 * This function is usually one of the first steps in preprocessing.
 *
 * It creates a binary image by thresholding each channel, and combining
 * each channel via bitwise_or. The threshold value is automatically calculated
 * using Otsu's method.
 *
 * @param src		the source image to binarize
 * @param dst		the destination image
 **/
void binarize( const Mat& src, Mat& dst )
{
	// threshold each channel
	vector<Mat> srcChans, dstChans;
	split( src, srcChans );
	for( int i = 0; i < (int)srcChans.size(); i++ )
	{
		dstChans.push_back( Mat() );
		threshold( srcChans[i], dstChans[i], 0, 255, THRESH_BINARY_INV|THRESH_OTSU );
	}

	// combine channels using bitwise_or
	dst = Mat::zeros( src.rows, src.cols, CV_8UC1 );
	for each( Mat chan in dstChans )
		dst = dst | chan;

	// close image by dilate followed by erode
	dilate( dst, dst, Mat() );
	erode( dst, dst, Mat() );

	// get rid of holes
	Mat holes = dst.clone();
	floodFill( holes, Point(0,0), Scalar(255,255,255) );
	holes = ~holes;
	dst = dst | holes;
}

/**
 * getBoundingRect
 *
 * This function finds the connected components,
 * determines which is largest (assumed to be the wasp body),
 * and returns a bounding rectangle for that contour.
 *
 * @param bin	the binary image
 * @return		the bounding rectangle
 **/
Rect getBoundingRect( const Mat& bin )
{
	assert( bin.type() == CV_8UC1 );

	Mat tmp = bin.clone();
	vector<vector<Point>> contours;
	findContours( tmp, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE );
	
	// find largest contour (assumed to be wasp body)
	double max = 0;
	int maxIdx = -1;
	for( int i = 0; i < (int)contours.size(); i++ )
		if( contourArea( contours[i] ) > max )
		{
			max = contourArea( contours[i] );
			maxIdx = i;
		}

	return boundingRect( contours[maxIdx] );
}

/**
 * calcColorWindow
 *
 * This method uses the idea of "spatially connected blocks" as
 * used in HOG features.
 *
 * Color features are calculated at blocks of length windowLength,
 * and separated by a distance windowStride. When windowStride is
 * less than windowLength, these blocks overlap and the resultant
 * feature vector contains information about spatial connectedness.
 *
 * @param src			the source image, preferrably cropped via getBoundingRect
 * @param windowLength	the length of a block. The height is simply the height of the image
 * @param windowStride	the distance to shift the block at each iteration
 * @return				the row feature vector for the color window or empty matrix if
 *							invalid parameters
 **/
Mat calcColorWindow( const Mat& src, int windowLength, int windowStride )
{
	if( windowLength > src.cols )
		return Mat();

	// number of blocks in the image
	// you can see that this is true if you work it out on pen and paper
	int numBlocks = ((src.cols - windowLength) / windowStride) + 1;

	// obtain color feature vector at each block and concatenate into one feature vector
	Mat colorWindowFeature( 1, colorFeatureLength * numBlocks, CV_32FC1 );

	for( int x = 0, startCol = 0; x <= src.cols-windowLength; 
		x += windowStride, startCol += colorFeatureLength )
	{
		Rect window( x, 0, windowLength, src.rows );

		Mat color_feature;
		calcColorFeature( src(window), color_feature );

		Mat header = colorWindowFeature.colRange( startCol, startCol + colorFeatureLength );
		color_feature.copyTo( header );
	}

	return colorWindowFeature;
}

/**
 * calcColorFeature
 *
 * This method retrieves the Hue/Saturation histogram feature
 * and stores it as a feature vector.
 *
 * This is based largely off the calcHist() example from:
 * <http://opencv.willowgarage.com/documentation/cpp/histograms.html>
 *
 * @param src	the color image to extract H-S histogram
 * @param dst	the feature vector of length hBins*sBins
 **/
void calcColorFeature( const Mat& src, Mat& dst, const Mat& mask )
{
	Mat hsv;
	MatND hist;

	/****************************************/
	// parameters for histogram extraction
	int histSize[] = {hbins, sbins};

	const float hranges[] = { 0, 180 };
	const float sranges[] = { 0, 256 };
	const float* ranges[] = { hranges, sranges };

	const int channels[] = {0, 1};
	/****************************************/
	cvtColor( src, hsv, CV_BGR2HSV );
	calcHist( &hsv, 1, channels, mask,
		hist, 2, histSize, ranges );

	// vectorize HS histogram
	dst = hist.reshape(0, 1);
}

/**
 * getNumLabels
 *
 * This function returns the total number of classes.
 * It does so by finding the maximum valued label,
 * and adding 1 (since labels are 0-indexed).
 *
 * @param labels	the matrix of labels
 * @return			the number of labels
 **/
int getNumLabels( const Mat& labels )
{
	double maxL;

	minMaxLoc( labels, 0, &maxL );
	return (int) maxL + 1; 
}

/**
 * mergeAllExcept
 *
 * This function takes in the vector of masks used for
 * training/testing. It combines K-1 of the sets (for training)
 * by bitwise or-ing the masks together, and leaves one
 * of the K sets out (for testing).
 *
 * @param indexSets		the partitioned set of masks
 * @param fold			the fold to be left out
 * @return				the merged set of masks for training data
 **/
Mat mergeAllExcept( const vector<Mat>& indexSets, int fold )
{
	// indexSets[0] is arbitrary; they all have the same number of rows
	Mat merged = Mat::zeros( indexSets[0].rows, 1, CV_8UC1 );

	// merge each mask except for the desired fold
	for( int i = 0; i < (int)indexSets.size(); i++ )
			if( i != fold ) 
				merged |= indexSets[i];

	return merged;
}

/**
 * splitData
 *
 * This function takes in the column vector of labels,
 * and returns a set of indices that represent "k" evenly
 * distributed parts.
 *
 * @param labels	the matrix of labels stored as a single column vector
 * @param k			the number of sets/folds desired
 * @return 			the vector of k evenly distributed masks
 **/
vector<Mat> splitData( const Mat& labels, int k )
{
	// out of bounds
	if( k < 1 || k > labels.rows ) 
		return vector<Mat>();

	// initialization
	vector<Mat> indexSets( k, Mat() );
	for( int i = 0; i < k; i++ )
		indexSets[i] = Mat::zeros( labels.rows, 1, CV_8UC1 );

	// for each fold, get every k-th feature and partition into one set
	for( int fold = 0; fold < k; fold++ )
		for( int i = fold; i < labels.rows; i+=k )
			indexSets[fold].at<uchar>(i, 0) = 1;

	return indexSets;
}

/**
 * onevsAllSVM_train
 *
 * Trains a vector of one-against-rest SVMs.
 * 
 * @param features		the matrix of features stored as row examples
 * @param labels		the corresponding column vector of class labels
 * @param varIdx		mask for selecting features
 * @param sampleIdx		mask for selecting samples
 * @param params		SVMparams to train each SVM with
 * @return				the vector of one-vs-all SVMs
 **/
vector<SVM> oneVsAllSVM_train( const Mat& features, const Mat& labels, 
	const Mat& varIdx, const Mat& sampleIdx, SVMParams params )
{
	int numLabels = getNumLabels( labels );

	/* create vector of each one-vs-all svm and corresponding training data */
	vector<SVM> svms( numLabels, SVM() );
	vector<Mat> svmLabels( numLabels, Mat() );
	
	/* initialize data */
	for( int i = 0; i < numLabels; i++ )
	{
		svms[i] = SVM();
		svmLabels[i] = labels.clone();
	}

	/* for each class, create one-vs-all labels */
	for( int i = 0; i < numLabels; i++ )
	{
		Mat currentClass = svmLabels[i];

		// change "self" label to 1, and all others to 0
		for( int j = 0; j < currentClass.rows; j++ )
			if( currentClass.at<float>(j, 0) == (float) i )
				currentClass.at<float>(j, 0) = 1.0;
			else
				currentClass.at<float>(j, 0) = -(1.0);
	}

	/* train each svm */
	for( int i = 0; i < numLabels; i++ )
		svms[i].train( features, svmLabels[i], varIdx, sampleIdx, params );

	return svms;
}

/**
 * onevsAllSVM_predict
 *
 * Takes in an example feature vector, and
 * predicts the class for that example.
 * Also returns a vector of probabilities corresponding
 * to the likelihood of each class for that feature vector.
 * 
 * @param svms				the vector of one-vs-all SVMs
 * @param feature			the feature vector to predict
 * @param confidenceVector	the vector containing how probable each class is
 * @return					the predicted class label
 **/
float oneVsAllSVM_predict( const vector<SVM>& svms, const Mat& feature, Mat& confidenceVector )
{
	Point maxLoc;

	confidenceVector = Mat( svms.size(), 1, CV_32FC1 );

	for( int i = 0; i < confidenceVector.rows; i++ )
	{
		confidenceVector.at<float>(i, 0) = 
			1 / (1 + exp(svms[i].predict( feature, true )));
	}

	minMaxLoc( confidenceVector, 0, 0, 0, &maxLoc );

	return (float) maxLoc.y;
}

/**
  * predictFromJointProb
  *
  * This function predicts a class by returning the index of
  * highest probability, by simple point-wise multiplication
  * of the conditionally independent posterior probabilities.
  *
  * Calculate joint probabilities for each class: 
  * P( class | color AND shape) =
  * P( class | color ) * P ( class | shape )
  *
  * @param colorProbs	the probabilties of a class based on color
  * @param hogProbs		the probabilities of a class based on HOG
  * @return				the most likely class
  **/
float predictFromJointProb( const Mat& colorProbs, const Mat& hogProbs, bool returnConfidence )
{
	Point maxLoc;
	double max;

	Mat jointProbs;
	multiply( colorProbs, hogProbs, jointProbs );
	minMaxLoc( jointProbs, 0, &max, 0, &maxLoc );
	
	if( returnConfidence )
		return (float) max;

	return (float) maxLoc.y;
}

/**
 * calculateAcc
 *
 * This function calculates the accuracy (numCorrect / numSamples)
 * for a given class, or for the entire dataset if no class
 * is specified.
 *
 * @param confusionMatrix	the confusion matrix
 * @param correct			pointer to float for numCorrect
 * @param total				pointer to float for numTotal
 * @param cl				the desired class for calculating class accuracy
 *						or -1 for calculating total accuracy
 * @return					the floating-point accuracy or (-1) if cl value
						is out of bounds
 **/
float calculateAcc( const Mat& confusionMatrix, float* correct, float* total, int cl )
{
	float numCorrect, numTotal;

	if ( cl < 0 )	// calculate total accuracy
	{	
		// sum up elements along the diagonal (correctly predicted)
		Mat diag = confusionMatrix.diag();
		numCorrect = (float) sum(diag)[0];
		numTotal = (float) sum(confusionMatrix)[0];
	} 
	else if ( cl < confusionMatrix.rows )	// calculate accuracy for row cl
	{	
		numCorrect = confusionMatrix.at<float>(cl, cl);
		Mat row = confusionMatrix.row(cl);
		numTotal = (float) sum(row)[0];
	}
	else	// cl out of bounds
	{	
		return (-1.0);
	}

	// return accuracy
	*correct = numCorrect;
	*total = numTotal;
	return (numCorrect / numTotal);
}

/**
 * printResults
 *
 * This function takes in the confusion matrix,
 * and displays the accuracies for each class,
 * along with the total accuracy and the confusion matrix itself.
 *
 * @param confusionMatrix	the confusion matrix
 **/
void printResults( const Mat& confusionMatrix )
{
	float correct, total;

	/* validate and print results */
	cout << "Individual class results:\n";

	for( int i = 0; i < confusionMatrix.rows; i++ )
	{
		float acc = calculateAcc( confusionMatrix, &correct, &total, i );
		cout << i << ": " << acc;
		cout << " (" << correct << "/" << total << ")" << endl;
	}

	/* display confusion matrix and results */
	cout << "\nConfusion Matrix:\n" << confusionMatrix << endl;
	cout << "Validation accuracy: " << calculateAcc( confusionMatrix, &correct, &total );
	cout << " (" << correct << "/" << total << ")" << endl;
}