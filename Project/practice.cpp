/******************************************************************************
 * File:			practice.cpp
 * Author:			Adrian Teng-Amnuay
 * Email:			atengamn@ucsd.edu, pumpadrian@gmail.com
 *
 * Description:		This program is open source, and was created to assist
 *					in automated Vespidae wasp recognition for TFRI, Taiwan.
 *
 *					This program contains functions that have already been
 *					implemented by the OpenCV library. By rewriting the
 *					methods, I gained a better understanding of the concepts
 *					used in computer vision and machine learning. These
 *					functions are deprecated, but can still be used to
 *					learn the ideas.
 *****************************************************************************/
#include "practice.h"
#include "misc.h"
using namespace cv;

/**
 * findCentroid
 *
 * This method finds the centroid via moments analysis.
 * Information on moments and centroids can be found here:
 * <http://en.wikipedia.org/wiki/Image_moment>
 *
 * @param src	the source image
 * @return		the centroid point
 **/
Point findCentroid( const Mat& src )
{
	Moments m = moments( src );
	return Point( (int)(m.m10/m.m00), (int)(m.m01/m.m00) );
}

/**
 * oneVsOneSVM_train
 *
 * Trains an adjacency matrix of one-vs-one SVM for each pair of classes
 * using the features and labels passed.
 *
 * @param trainFeatures	the matrix of features to train the model
 * @param trainLabels		the corresponding column vector of ground truth labels
 * @param numLabels			the total number of classes
 * @return					the adjacency matrix of SVMs as a 2-D array of SVMs
 **/
vector<vector<SVM>> oneVsOneSVM_train( const Mat& trainFeatures, const Mat& trainLabels, int numLabels, SVMParams params )
{
	/* create 2-D array of one-vs-one SVMs and appropriate training data for each*/
	vector<vector<SVM>> svms( numLabels, vector<SVM>(numLabels, SVM()) );
	vector<vector<Mat>> svm_data( numLabels, vector<Mat>(numLabels, Mat()) );
	vector<vector<Mat>> svm_labels(	numLabels, vector<Mat>(numLabels, Mat()) );
	
	/* initialize data */
	for( int i = 0; i < numLabels; i++ )
	{
		for( int j = 0; j < numLabels; j++ )
		{
			svm_data[i][j]	 = Mat( 0, trainFeatures.cols, CV_32FC1 );
			svm_labels[i][j] = Mat( 0, 1, CV_32FC1 );
			
		}
	}

	/* organize data into 2-D array of one-vs-one features/labels */
	for( int i = 0; i < trainFeatures.rows; i++ )
	{
		int curLabel = (int) trainLabels.at<float>(i, 0);
		Mat curFeature = trainFeatures.row(i);
		for( int other = 0; other < numLabels; other++ )
		{
			if( curLabel == other) continue; // skip diagonals

			Mat oneOne = Mat::ones( 1, 1, CV_32FC1 );
			Mat oneZero = Mat::zeros( 1, 1, CV_32FC1 );

			svm_data[curLabel][other].push_back( curFeature );
			svm_labels[curLabel][other].push_back( oneOne );

			svm_data[other][curLabel].push_back( curFeature );
			svm_labels[other][curLabel].push_back( oneZero );
		}
	}
	
	/* train each svm */
	for( int i = 0; i < numLabels; i++ ) 
	{
		for( int j = 0; j < numLabels; j++ ) 
		{
			if( i == j ) continue; // skip diagonals
			svms[i][j].train( svm_data[i][j], svm_labels[i][j], Mat(), Mat(), params );
		}
	}

	return svms;
}

/**
 * oneVsOneSVM_predict
 *
 * Predicts a class from the passed in feature by
 * predicting the class for each one-vs-one SVM,
 * and returning the result with the highest number of votes.
 *
 * @param svms		the 2-D array of trained one-vs-one SVMs
 * @param feature	the feature to predict
 * @return			the predicted label
 **/
float oneVsOneSVM_predict( const vector<vector<SVM>>& svms, const Mat& feature )
{
	// initialize
	int mostVotes = 0;
	float bestClass = -1;

	for( int i = 0; i < (int)svms.size(); i++ )
	{
		int numVotes = 0;

		// count the number of times class this is predicted
		for( int j = 0; j < (int)svms[i].size(); j++ )
		{
			if( i == j ) continue;
			if( svms[i][j].predict( feature ) == 1.0 )
				numVotes++;
		}

		// if better than current, replace
		if( numVotes > mostVotes ) {
			mostVotes = numVotes;
			bestClass = (float) i;
		}
	}

	return bestClass;
}

/**
 * oneVsAllSVM_train
 *
 * Trains the array of one-vs-all SVM for each class
 * using the features and labels passed.
 *
 * @param trainFeatures	the matrix of features to train the model
 * @param trainLabels		the corresponding column vector of ground truth labels
 * @param numLabels			the total number of classes
 * @return					the vector of one-vs-all SVMs
 **/
vector<SVM> oneVsAllSVM_train( const Mat& trainFeatures, const Mat& trainLabels, int numLabels, SVMParams params )
{
	/* create vector of each one-vs-all svm and corresponding training data */
	vector<SVM> svms( numLabels, SVM() );
	vector<Mat> svm_data( numLabels, Mat() );
	vector<Mat> svm_labels( numLabels, Mat() );
	
	/* initialize data */
	for( int i = 0; i < numLabels; i++ )
	{
		svm_data[i] = Mat( 0, trainFeatures.cols, CV_32FC1 );
		svm_labels[i] = Mat( 0, 1, CV_32FC1 );
	}

	/* organize data into one-vs-all features/labels */
	for( int i = 0; i < trainFeatures.rows; i++ )
	{
		int curLabel = (int) trainLabels.at<float>(i, 0);
		Mat curFeature = trainFeatures.row(i);

		Mat oneOne = Mat::ones( 1, 1, CV_32FC1 );
		Mat oneZero = Mat::zeros( 1, 1, CV_32FC1 );

		for( int other = 0; other < numLabels; other++ )
			if( curLabel == other )	// "one"
			{
				svm_data[curLabel].push_back( curFeature );
				svm_labels[curLabel].push_back( oneOne );
			} 
			else	// "all"
			{
				svm_data[other].push_back( curFeature );
				svm_labels[other].push_back( oneZero );
			}
	}

	/* train each svm */
	for( int i = 0; i < numLabels; i++ )
		svms[i].train( svm_data[i], svm_labels[i], Mat(), Mat(), params );

	return svms;
}

/**
 * oneVsAllSVM_predict
 *
 * Predicts a class from the passed in feature by
 * predicting the class for each one-vs-all SVM,
 * and returning the class with the lowest decision
 * function value.
 *
 * @param svms		the vector of trained one-vs-all SVMs
 * @param feature	the feature to predict
 * @return			the predicted label
 **/
float oneVsAllSVM_predict( const vector<SVM>& svms, const Mat& feature )
{
	// return whichever class has the lowest DFval
	float minDFval = (float) INT_MAX;
	float bestClass = -1;
	for( int i = 0; i < (int)svms.size(); i++ )
	{
		/*cout << svms[i].predict( feature ) << endl;
		cout << i << ": " << svms[i].predict( feature, true ) << endl;*/
		// if better than current, replace
		if( svms[i].predict( feature, true ) < minDFval )
		{
			minDFval = svms[i].predict( feature, true );
			bestClass = (float) i;
		}
	}

	return bestClass;
}

Mat drawHog( const Mat& src, int len )
{
	Mat gray;
	
	cvtColor( src, gray, CV_BGR2GRAY );
	vector<Mat> integrals = calculateIntegralHOG( gray );

	Mat hogdraw( src.rows, src.cols, CV_32FC1 );

	Mat hog_cell(1,9,CV_32FC1);

	for( int r = 0; r < src.rows-len; r+=len )
	{
		for( int c = 0; c < src.cols-len; c+=len )
		{
			calculateHOG_rect( Rect(c,r,len,len), hog_cell, integrals, 4 );

			//get max angle
			float max = 0; int ind = 0;
			for( int i = 0; i < 9; i++ )
				if( hog_cell.at<float>(0,i) > max ) { 
					max = hog_cell.at<float>(0,i);
					ind = i;
				}

				float angle = (float) ((ind*20) + 20);

				RotatedRect roi(	Point2f( (float)(c+len/2),(float)(r+len/2)),
									Size((int)(max*len),2), angle );

				ellipse( hogdraw, roi, CV_RGB(255,255,255) );
		}
	} 
	return hogdraw;
}

// HOG-related functions
/**
 *The following code is based on the code found here:
 *<http://smsoftdev-solutions.blogspot.com/2009/08/integral-histogram-for-fast-calculation.html>
 *<http://smsoftdev-solutions.blogspot.com/2009/10/object-detection-using-opencv-ii.html>
**/
vector<Mat> calculateIntegralHOG(const Mat& src)
{
	assert( src.channels() == 1 );	// gray-scale
	
	Mat xsobel, ysobel;
	Sobel( src, xsobel, CV_8UC1, 1, 0, 1 );
	Sobel( src, ysobel, CV_8UC1, 0, 1, 1 );
	
	vector<Mat> bins( 9, Mat() ), integrals( 9, Mat() );
	for( int i = 0; i < 9; i++ )
	{
		bins[i] = Mat::zeros( src.rows, src.cols, CV_32FC1 );
		integrals[i] = Mat( src.rows+1, src.cols+1, CV_64FC1 );
	}

	float temp_gradient, temp_magnitude;
	for( int r = 0; r < src.rows; r++ )
	{
		for( int c = 0; c < src.cols; c++ )
		{
			float xSob = (float) xsobel.at<uchar>(r,c);
			float ySob = (float) ysobel.at<uchar>(r,c);

			if ( xSob == 0 )	temp_gradient = saturate_cast<float>( ((atan(ySob / (xSob + 0.00001))) * (180/ CV_PI)) + 90 );
			else				temp_gradient = saturate_cast<float>( ((atan(ySob / xSob)) * (180 / CV_PI)) + 90 );
			
			temp_magnitude = sqrt((xSob * xSob) + (ySob * ySob));

			if		( temp_gradient <= 20 )		bins[0].at<float>(r,c) = temp_magnitude;
			else if ( temp_gradient <= 40 )		bins[1].at<float>(r,c) = temp_magnitude;
			else if ( temp_gradient <= 60 )		bins[2].at<float>(r,c) = temp_magnitude;
			else if ( temp_gradient <= 80 )		bins[3].at<float>(r,c) = temp_magnitude;
			else if ( temp_gradient <= 100 )	bins[4].at<float>(r,c) = temp_magnitude;
			else if ( temp_gradient <= 120 )	bins[5].at<float>(r,c) = temp_magnitude;
			else if ( temp_gradient <= 140 )	bins[6].at<float>(r,c) = temp_magnitude;
			else if ( temp_gradient <= 160 )	bins[7].at<float>(r,c) = temp_magnitude;
			else								bins[8].at<float>(r,c) = temp_magnitude;
		}
	}

	for( int i = 0; i < 9; i++) 
		integral( bins[i], integrals[i] );

	return integrals;
}

void calculateHOG_rect( Rect cell, Mat& hog_cell, const vector<Mat>& integrals, int normalization ) 
{
	for( int i = 0; i < 9; i++ )
	{
		double a = integrals[i].at<double>(cell.y, cell.x);
		double b = integrals[i].at<double>(cell.y + cell.height, cell.x + cell.width);
		double c = integrals[i].at<double>(cell.y, cell.x + cell.width);
		double d = integrals[i].at<double>(cell.y + cell.height, cell.x);

		hog_cell.at<float>(0,i) = (float) ((a + b) - (c + d));
	}

	/*Normalize the matrix*/
	if (normalization != -1)
		normalize( hog_cell, hog_cell, 1, 0, normalization );
	
}

void calculateHOG_block(Rect block, Mat& hog_block, const vector<Mat>& integrals, Size cell, int normalization) 
{
	int startcol = 0;
	for (int cell_start_y = block.y; cell_start_y <= block.y + block.height - cell.height; cell_start_y += cell.height) 
	{
		for (int cell_start_x = block.x; cell_start_x <= block.x + block.width - cell.width; cell_start_x += cell.width) 
		{
			
			Mat vector_cell = hog_block.colRange( startcol, startcol + 9 );

			calculateHOG_rect(Rect(cell_start_x,
				cell_start_y, cell.width, cell.height), 
				vector_cell, integrals, -1);

			startcol += 9;
		}
	}

	/*Normalize the matrix*/
	if (normalization != -1)
		normalize( hog_block, hog_block, 1, 0, normalization );
}

Mat calculateHOG_window(const vector<Mat>& integrals, Rect window, int normalization) 
{

	/*A cell size of 8x8 pixels is considered and each
	block is divided into 2x2 such cells (i.e. the block
	is 16x16 pixels). So a 64x128 pixels window would be
	divided into 7x15 overlapping blocks*/ 

	int cell_width = 8, cell_height = 8;
	int block_width = 2, block_height = 2;

	/* The length of the feature vector for a cell is
	9(since no. of bins is 9), for block it would  be
	9*(no. of cells in the block) = 9*4 = 36. And the
	length of the feature vector for a window would be
	36*(no. of blocks in the window */

	Mat window_feature_vector(1,
		((((window.width - cell_width * block_width)
		/ cell_width) + 1) * (((window.height -
		cell_height * block_height) / cell_height)
		+ 1)) * 36, CV_32FC1);

	Mat vector_block;
	int startcol = 0;
	for (	int block_start_y = window.y; 
			block_start_y <= window.y + window.height - cell_height * block_height; 
			block_start_y += cell_height)
	{
		for (	int block_start_x = window.x; 
				block_start_x <= window.x + window.width - cell_width * block_width; 
				block_start_x += cell_width)
		{
			vector_block = window_feature_vector.colRange( startcol, startcol + 36 );

			calculateHOG_block(Rect(block_start_x,
				block_start_y, cell_width * block_width, cell_height
				* block_height), vector_block, integrals, Size(
				cell_width, cell_height), normalization);

			startcol += 36;
		}
	}
	return (window_feature_vector);
}