/******************************************************************************
 * File:			wasp_classifier.cpp
 * Author:			Adrian Teng-Amnuay
 * Email:			atengamn@ucsd.edu, pumpadrian@gmail.com
 *
 * Description:		This program is open source, and was created to assist
 *					in automated Vespidae wasp recognition for TFRI, Taiwan.
 *					It makes extensive use of the OpenCV 2.3 library.
 *
 *					The program assumes a file defined by FILEPATHS_AND_LABELS
 *					that contains the paths to all the images used for training
 *					or testing (this file is generated through the python 
 *					script). 
 *
 *					For training the model, color histograms and 
 *					HOG descriptors are extracted from the images, and then 
 *					are the number of features are reduced using PCA.
 *					A multiclass SVM is then trained using these features.
 *					Validation results are displayed, and the model, along
 *					with PCA basis vectors and mean are saved in .xml files.
 *
 *					For testing the model, the SVM model and principal
 *					component vectors are loaded. Then the test images are
 *					automatically labeled and the results are stored to
 *					the file defined in macro CLASS_LABELS_OUT.					
 *****************************************************************************/
#include <cv.h>
#include <ml.h>
#include <highgui.h>

#include <fstream>
#include <string>
#include <sstream>
#include <iterator>

#include "misc_recognition.h"

using namespace cv;
using namespace std;

// for saving and loading model
#define SVM_FILENAME			"SVM_data.xml"
#define COLOR_SVM				"colorSVM"
#define HOG_SVM					"hogSVM"
#define PCA_FILENAME			"PCA_data.xml"
#define COLOR_BASISVECTORS		"colorBasisVectors"
#define COLOR_MEAN				"colorMean"
#define HOG_BASISVECTORS		"hogBasisVectors"
#define HOG_MEAN				"hogMean"
#define NUM_LABELS				"numLabels"

// python script and file IO
#define PY_SCRIPT				"get_paths.py"
#define FILEPATHS_AND_LABELS	"filepaths_and_labels.txt"
#define CLASS_LABELS_OUT		"class_labels.txt"

// image processing params
#define avgRectWidth			117
#define avgRectHeight			34

// miscellaneous
#define PLUS_OR_MINUS			241

/* function prototypes */
void testModel();
void trainModel();
void extractOnlyFeatures( Mat& colorFeatures, Mat& hogFeatures, vector<string>& fileNames );
void extractFeaturesAndLabels( Mat& colorFeatures, Mat& hogFeatures, Mat& labels );
Mat preProcess( const Mat& src );
Mat getColorFeature( const Mat& src );
Mat getHOGFeature( const Mat& src);

void usage(string filename)
{
	string programName = filename.substr( filename.find_last_of("/\\")+1 );

	cerr << "Usage: " << programName << " <mode> <dir>\n\n";
	cerr << "<mode> = 0 for training, or 1 for testing.\n";
	cerr << "<dir> is the name of the directory.\n\n";
	cerr << "For training, <dir> contains subdirectory of images for each species.\n";
	cerr << "For testing, <dir> simply contains the images.\n";

	exit( EXIT_FAILURE );
}

/** 
 * main method
 *
 * Currently, the program assumes a file named "filepaths_and_labels.txt"
 * that contains the paths to the images.
 * 
 * The program loads this file, and begins reading in the images specified by
 * the text file. A feature vector is extracted from each image (described in function headers)
 * and these features are organized into a matrix (each example as a row vector).
 * The corresponding class labels are organized as a single column vector if program is
 * in training mode.
 *
 * For training, the model is trained and cross-validated, then saved to file.
 * For testing, the model is loaded and the images are predicted.
 *
 * Currently multiClass SVM is default classifier and shows best performance.
 * Other models available for testing are k-Nearest Neighbor and Decision Tree.
 *
 * Results are then displayed to stdout via confusion matrices and accuracy percentages.
 **/
int main( int argc, char** argv )
{
	if( argc != 3 )
		usage( argv[0] );

	int mode = atoi( argv[1] );
	if( mode != 0 && mode != 1 )
		usage( argv[0] );

	// run python script
	string command = "python " 
		+ string(PY_SCRIPT) + " " 
		+ string(argv[1]) + " " 
		+ string(argv[2]);

	if( system(command.c_str()) )
		exit( EXIT_FAILURE );

	if( mode == 0 )	
		trainModel();
	else
		testModel();
	
	return 0;
}

void testModel()
{
	cout << "[Loading model from XML files]\n";

	if ( !ifstream(PCA_FILENAME) ) {
		cerr << "File not found: " << PCA_FILENAME << ", ";
		cerr << "please train model first.\n";
		exit( EXIT_FAILURE );
	}

	if ( !ifstream(SVM_FILENAME) ) {
		cerr << "File not found: " << SVM_FILENAME << ", ";
		cerr << "please train model first.\n";
		exit( EXIT_FAILURE );
	}

	/* open file storage */
	FileStorage fs( PCA_FILENAME, FileStorage::READ );

	int numLabels = (int) fs[NUM_LABELS];	// number of classes

	/* allocate SVM vectors */
	vector<SVM> colorSVMs( numLabels, SVM() );
	vector<SVM> hogSVMs( numLabels, SVM() );
	
	/* load each SVM from file */
	for( int i = 0; i < numLabels; i++ )
	{
		stringstream out; out << i;
		colorSVMs[i].load( SVM_FILENAME, string(COLOR_SVM + out.str()).c_str() );
		hogSVMs[i].load( SVM_FILENAME, string(HOG_SVM + out.str()).c_str() );
	}
	
	/* load PCA from file */
	PCA pcaColor, pcaHOG;
	fs[COLOR_BASISVECTORS] >> pcaColor.eigenvectors;
	fs[COLOR_MEAN] >> pcaColor.mean;
	fs[HOG_BASISVECTORS] >> pcaHOG.eigenvectors;
	fs[HOG_MEAN] >> pcaHOG.mean;
	
	cout << "Labeling data...";

	/* extract features and filenames */
	Mat colorFeatures, hogFeatures;
	vector<string> fileNames;
	extractOnlyFeatures( colorFeatures, hogFeatures, fileNames );

	/* reduce dimensions */
	colorFeatures = pcaColor.project( colorFeatures );
	hogFeatures = pcaHOG.project( hogFeatures );

	/* open file stream for writing */
	ofstream ofs( CLASS_LABELS_OUT );
	if (!ofs) {
		cerr << "Could not open file for writing." << endl;
		exit( EXIT_FAILURE );
	}

	// write number of samples
	ofs << colorFeatures.rows << endl;

	/* predict class for testing data */
	for( int i = 0; i < colorFeatures.rows; i++ )
	{ 
		string fileName = fileNames[i];

		Mat colorProbs, hogProbs;
		oneVsAllSVM_predict( colorSVMs, colorFeatures.row(i), colorProbs );
		oneVsAllSVM_predict( hogSVMs, hogFeatures.row(i), hogProbs );
		
		/* predict class from cond. independent posterior probabilities */
		float predicted = predictFromJointProb( colorProbs, hogProbs );

		/* write prediction to file */
		ofs << fileName << " " << predicted << endl;
	}

	cout << "\b\b\b done!\nClass labels written to \"" << CLASS_LABELS_OUT << "\"\n";
}

void trainModel()
{
	/* extract features and labels */
	cout << "[Training model]\nExtracting features...";
	Mat colorFeatures, hogFeatures, labels;
	extractFeaturesAndLabels( colorFeatures, hogFeatures, labels );
	cout << "\b\b\b done!\n";

	int numLabels = getNumLabels( labels );	// number of classes

	/* dimensionality reduction using PCA */
	cout << "Reducing " << colorFeatures.cols << " color features to...";
	PCA pcaColor( colorFeatures, Mat(), CV_PCA_DATA_AS_ROW, 200 );
	colorFeatures = pcaColor.project( colorFeatures );
	cout << "\b\b\b " << colorFeatures.cols << ". done!\n";

	cout << "Reducing " << hogFeatures.cols << " HOG features to...";
	PCA pcaHOG( hogFeatures, Mat(), CV_PCA_DATA_AS_ROW, 300 );
	hogFeatures = pcaHOG.project( hogFeatures );
	cout << "\b\b\b " << hogFeatures.cols << ". done!\n";
	
	/** Partition data into numFolds separate sets, eg.
	  * K folds for cross-validation
	  *	2 folds for hold-out validation
	  *	labels.rows folds for leave-one-out validation
	  *	Once sets are partitioned, mix and match as necessary 
	  **/
	int numFolds = 10;
	vector<Mat> indexSets =	splitData( labels, numFolds );

	// SVM params
	SVMParams params;
	params.svm_type=SVM::C_SVC;
	params.kernel_type=SVM::LINEAR;

	/* allocate structures for collecting statistical data */
	float correct, total;
	Mat confusionMatrix = Mat::zeros( numLabels, numLabels, CV_32FC1 );
	Mat foldMeans = Mat( numFolds, 1, CV_32FC1 );		// to find stddev

	cout << "Cross-validating using " << numFolds << " folds\n\n";
	for( int fold = 0; fold < numFolds; fold++ )
	{
		/* partition data into training and testing sets */
		Mat testIndices = indexSets[fold];
		Mat trainIndices = mergeAllExcept( indexSets, fold );

		/* train classifier */
		vector<SVM> colorSVMs 
			= oneVsAllSVM_train( colorFeatures, labels, Mat(), trainIndices, params );
		vector<SVM> hogSVMs 
			= oneVsAllSVM_train( hogFeatures, labels, Mat(), trainIndices, params );

		//SVM svm( colorFeatures, labels, Mat(), trainIndices, params );
		//KNearest knn( colorFeatures, labels, trainIndices );
		//DecisionTree dtree; //CvDTreeParams treeParams( INT_MAX, 1, (0.01f), true, numLabels, 10, true, true, 0 );
		//dtree.train( trainFeatures, CV_ROW_SAMPLE, trainLabels, Mat(), Mat(), Mat(), Mat(), treeParams );

		/* allocate confusion mat for each fold */
		Mat foldConfMat = Mat::zeros( numLabels, numLabels, CV_32FC1 );	
		
		/* predict class for testing data */
		for( int i = 0; i < labels.rows; i++ )
		{
			float predicted, actual;

			if( testIndices.at<uchar>(i, 0) == 0 ) // only predict test examples
				continue;

			//predicted = knn.find_nearest( colorFeatures.row(i), 2 );
			//predicted = dtree.predict( colorFeatures.row(i) )->value;
			//predicted = svm.predict( colorFeatures.row(i) );
			
			Mat colorProbs, hogProbs;
			oneVsAllSVM_predict( colorSVMs, colorFeatures.row(i), colorProbs );
			oneVsAllSVM_predict( hogSVMs, hogFeatures.row(i), hogProbs );

			/* predict class from cond. independent posterior probabilities */
			predicted = predictFromJointProb( colorProbs, hogProbs );
			actual = labels.at<float>(i, 0);

			//cout << colorProbs.at<float>(predicted,0) << " ";
			//cout << hogProbs.at<float>(predicted, 0) << " ";
			//cout << predictFromJointProb( colorProbs, hogProbs, true ) << endl;
			//cout << "predicted: " << predicted << "actual: " << actual << endl;
			//cin.ignore(1);
		
			foldConfMat.at<float>((int)actual, (int)predicted) ++;
		}

		/* average results for each fold */
		foldMeans.at<float>(fold, 0) = calculateAcc( foldConfMat, &correct, &total );
		confusionMatrix += foldConfMat;
	}
	
	printResults( confusionMatrix );
	Scalar stddev; meanStdDev( foldMeans, Scalar(), stddev );
	cout << "Standard Deviation: " << (char) PLUS_OR_MINUS << stddev[0] << endl;

	cout << "\n[Saving model to XML files]\n";

	/* re-train on full data-set */
	vector<SVM> colorSVMs = oneVsAllSVM_train( colorFeatures, labels, Mat(), Mat(), params );
	vector<SVM> hogSVMs = oneVsAllSVM_train( hogFeatures, labels, Mat(), Mat(), params );

	/* open file storage */
	CvFileStorage* cvfs = cvOpenFileStorage( SVM_FILENAME, 0, CV_STORAGE_WRITE );

	/* save each one-vs-all SVM */
	for( int i = 0; i < numLabels; i++ )
	{
		stringstream out; out << i;
		colorSVMs[i].write( cvfs, string(COLOR_SVM + out.str()).c_str() );
		hogSVMs[i].write( cvfs, string(HOG_SVM + out.str()).c_str() );
	}
	cvReleaseFileStorage( &cvfs );	// close file storage
	
	/* save PCA vectors and mean */
	FileStorage fs( PCA_FILENAME, FileStorage::WRITE );
	fs << COLOR_BASISVECTORS << pcaColor.eigenvectors << COLOR_MEAN << pcaColor.mean;
	fs << HOG_BASISVECTORS << pcaHOG.eigenvectors << HOG_MEAN << pcaHOG.mean;
	fs << NUM_LABELS << numLabels;		// also save nmber of classes
	fs.release();
	cout << "Done!\n";
}

/**
 * extractOnlyFeatures
 *
 * This method is similar to extractFeaturesAndLabels, except that
 * there are no labels for testing. It extracts the features of
 * the images in "filepaths_and_labels.txt" along with the names
 * of the images corresponding to the features.
 *
 * @param features	the features to be extracted
 * @param fileNames	the filenames associated with those features
 **/
void extractOnlyFeatures( Mat& colorFeatures, Mat& hogFeatures, vector<string>& fileNames )
{
	ifstream ifs( FILEPATHS_AND_LABELS );
	if (!ifs) {
		cerr << "File not found: " << FILEPATHS_AND_LABELS << endl;
		exit( EXIT_FAILURE );
	}

	/* get number of samples */
	string line;
	getline( ifs, line ); 	
	int numSamples = atoi( line.c_str() );
	
	/* collect features for every image */
	for( int i = 0; i < numSamples; i++ )
	{
		string filePath;
		getline( ifs, filePath );

		size_t found = filePath.find_last_of( "/\\" );
		fileNames.push_back( filePath.substr(found+1) );
				
		Mat img = imread( filePath ); 
		Mat proc = preProcess( img );
		Mat colorFeat = getColorFeature( proc );
		Mat hogFeat = getHOGFeature( proc );

		/* allocate space for feature matrix */
		if( i == 0 ) 
		{
			colorFeatures = Mat( numSamples, colorFeat.cols, CV_32FC1 );
			hogFeatures = Mat( numSamples, hogFeat.cols, CV_32FC1 );
		}
		
		colorFeat.copyTo( colorFeatures.row(i) );
		hogFeat.copyTo( hogFeatures.row(i) );
	}
}

/**
 * extractFeaturesAndLabels
 * 
 * This method loads the images and ground truth labels specified
 * by "filepaths_and_labels.txt" and extracts the features,
 * storing them into a matrix as row examples.
 *
 * @param colorFeatures	the matrix of color feature vectors stored as row examples
 * @param hogFeatures	the matrix of HOG feature vectors stored as row examples
 * @param labels		the matrix of labels, stored as one column vector
 **/
void extractFeaturesAndLabels( Mat& colorFeatures, Mat& hogFeatures, Mat& labels )
{
	ifstream ifs( FILEPATHS_AND_LABELS );
	if (!ifs) {
		cerr << "File not found: filepaths_and_labels.txt" << endl;
		exit( EXIT_FAILURE );
	}

	string line;
	getline( ifs, line ); 	
	int numSamples = atoi( line.c_str() );

	labels = Mat( numSamples, 1, CV_32FC1 );	
	
	/* collect features for every image */
	for( int i = 0; i < numSamples; i++ )
	{
		string filePath, classLabelString;
		getline( ifs, filePath, '@' );
		getline( ifs, classLabelString );

		labels.at<float>(i, 0) = 
			(float) atoi( classLabelString.c_str() );	
				
		Mat img = imread( filePath );
		Mat proc = preProcess( img );
		Mat colorFeat = getColorFeature( proc );
		Mat hogFeat = getHOGFeature( proc );
		
		/* allocate space for feature matrix */
		if( i == 0 ) 
		{
			colorFeatures = Mat( numSamples, colorFeat.cols, CV_32FC1 );
			hogFeatures = Mat( numSamples, hogFeat.cols, CV_32FC1 );
		}
		
		colorFeat.copyTo( colorFeatures.row(i) );
		hogFeat.copyTo( hogFeatures.row(i) );
	}	
}

/**
 * preProcess
 * 
 * This function defines the image pre-processing step.
 * The image is first blurred to smooth out noise and edges.
 * This helps in extracting a clean binary image.
 * A bounding rectangle is then fit over the wasp in the
 * binary image, and the rectangle is cropped from the image.
 * The rectangle is then resized and returned.
 *
 * @param src	the source image
 * @return		the pre-processed image
 **/
Mat preProcess( const Mat& src )
{
		Mat pyrdown, blur, bin, proc; 
		pyrDown( src, pyrdown );
		
		//medianBlur( pyrdown, blur, 3 );
		GaussianBlur( pyrdown, blur, Size(7,7), 2 );	// blur to remove noise
		binarize( blur, bin );							// binary image for region of interest
		
		/* find region of interest and normalize image sizes */
		Rect roi = getBoundingRect( bin );
		resize( pyrdown(roi), proc, Size(avgRectWidth, avgRectHeight) );

		return proc;
}

/**
 * getColorFeature
 * 
 * This function calculates the color feature from the image
 * by using overlapping blocks to calculate individual
 * Hue/Saturation histograms, then concatenating the individual
 * feature bins together.
 *
 * @param src	the source image
 * @return		the feature vector
 **/
Mat getColorFeature( const Mat& src )
{
	/* calculate color features (somewhat optimal after several experiments) */
	int windowLength = src.cols/4;
	int windowStride = src.cols/8;
	Mat colorWindowFeature = calcColorWindow( src, windowLength, windowStride );

	//return feature vector;
	return colorWindowFeature;
}

/**
 * getHOGFeature
 * 
 * This function calculates the HOG Window feature from the image
 * using the OpenCV HOGDescriptor class.
 *
 * @param src	the source image
 * @return		the feature vector
 **/
Mat getHOGFeature( const Mat& src )
{
	/* hog parameters (somewhat optimal after several experiments) */
	vector<float> hogDescriptors;
	int numBins = 7;
	Size cellSize(8, 8);
	Size blockSize(cellSize * 3);
	Size windowSize(	(avgRectWidth / blockSize.width) * blockSize.width,
						(avgRectHeight / blockSize.height) * blockSize.height );
		
	/* calculate hog features */
	HOGDescriptor hog( windowSize, blockSize, cellSize, cellSize, numBins );
	hog.compute( src, hogDescriptors );		
	Mat hogFeature( hogDescriptors );		// create matrix from vector
	hogFeature = hogFeature.reshape(0, 1);	// reshape to a row vector
	
	return hogFeature.clone();
}