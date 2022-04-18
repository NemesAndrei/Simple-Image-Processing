// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <queue>
#include <random>
using namespace std;



void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat_ <uchar> src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat_ <uchar> dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src(i,j);
				uchar neg = 255 - val;
				dst(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat_<Vec3b> src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat_ <uchar> dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

//LABORATOR 16.02.2022
void testAdditiveFactor(int k)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat_ <uchar> src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat_ <uchar> dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if ((src(i, j) + k <= 255) && (src(i, j) + k >= 0)) {
					int val = src(i, j) + k;
					dst(i, j) = val;
				}
				else {
					if (src(i, j) + k >= 255) {
						dst(i, j) = 255;
					}
					else {
						dst(i, j) = 0;
					}
				}
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testMultiplicativeFactor(float k)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat_ <uchar> src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat_ <uchar> dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				float val = src(i, j) * k;
				if (val <= 255) {
					dst(i, j) = (uchar)val;
				}
				else {
					dst(i, j) = 255;
				}
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		
		imshow("input image", src);
		imshow("negative image", dst);
		
		waitKey();
		imwrite("hello.bmp", dst); //writes the destination to file
	}
}

void testSquaresImage() 
{
	Mat_ <Vec3b> imgSquares(256, 256);
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 256; j++) {
			if (i <= 127 && j <= 127) {
				Vec3b pixel = imgSquares(i, j);
				pixel[0] = 255;
				pixel[1] = 255;
				pixel[2] = 255;
				imgSquares(i, j) = pixel;
			}
			if (i > 127 && j <= 127) {
				Vec3b pixel = imgSquares(i, j);
				pixel[0] = 0;
				pixel[1] = 0;
				pixel[2] = 255;
				imgSquares(i, j) = pixel;
			}
			if (i <= 127 && j > 127) {
				Vec3b pixel = imgSquares(i, j);
				pixel[0] = 0;
				pixel[1] = 255;
				pixel[2] = 0;
				imgSquares(i, j) = pixel;
			}
			if (i >127 && j > 127) {
				Vec3b pixel = imgSquares(i, j);
				pixel[0] = 0;
				pixel[1] = 255;
				pixel[2] = 255;
				imgSquares(i, j) = pixel;
			}
		}
	}
	imshow("Squares", imgSquares);
	waitKey();
	imwrite("squares.bmp", imgSquares);
}

void testInverse() {
	float vals[9] = { 1, 2, 2, 3, 4, 1, 5, 2, 9 };
	Mat_ <float> M = Mat (3,3,CV_32FC1,vals);
	Mat_ <float> invM = M.inv();
	std::cout << M << std::endl;
	std::cout << std::endl;
	std::cout << invM << std::endl;
	std::cout << invM * M << std::endl;
}

//LABORATOR 23.02

void image2RGB2Grey() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_ <Vec3b> src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat_ <uchar> rMat = Mat(height, width, CV_8UC1);
		Mat_ <uchar> gMat = Mat(height, width, CV_8UC1);
		Mat_ <uchar> bMat = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				Vec3b pixel = src(i, j);
				rMat(i, j) = pixel[2];
				gMat(i, j) = pixel[1];
				bMat(i, j) = pixel[0];
			}
		}
		imshow("original", src);
		imshow("red", rMat);
		imshow("green", gMat);
		imshow("blue", bMat);
		waitKey();
	}
}

void testColorToGray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<Vec3b> src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat_ <uchar> dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testGray2BW() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, 0);

		int height = src.rows;
		int width = src.cols;
		int threshold;
		scanf("%d", &threshold);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src(i, j) < threshold) {
					src(i, j) = 0;
				}
				else {
					src(i, j) = 255;
				}
			}
		}

		imshow("input image", src);
		waitKey();
	}
}

void testRGB2HSV() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<Vec3b> src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat_ <uchar> H = Mat(height, width, CV_8UC1);
		Mat_ <uchar> S = Mat(height, width, CV_8UC1);
		Mat_ <uchar> V = Mat(height, width, CV_8UC1);
		Mat_ <Vec3b> resultHSV(height,width);
		Mat_<Vec3b> rgbResult(height, width);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src(i, j);
				float b = (float)v3[0]/255;
				float g = (float)v3[1]/255;
				float r = (float)v3[2]/255;

				float M = max(max(r, g), max(g, b));
				float m = min(min(r, g), min(g, b));
				float C = M - m;

				float Haux;
				float Saux;
				float Vaux;
				Vaux = M;
				if (Vaux != 0) {
					Saux = C/Vaux;
				}
				else {
					Saux = 0;
				}
				if (C != 0) {
					if (M == r) {
						Haux = 60 * (g - b) / C;
					}
					if (M == g) {
						Haux = 120 + 60 * (b - r) / C;
					}
					if (M == b) {
						Haux = 240 + 60 * (r - g) / C;
					}
				}
				else {
					Haux = 0;
				}
				if (Haux < 0) {
					Haux = Haux + 360;
				}
				H(i, j) = (uchar)(Haux * 255 / 360);
				S(i, j) = (uchar)(Saux * 255);
				V(i, j) = (uchar)(Vaux * 255);
				Vec3b point;
				point[0] = H(i, j);
				point[1] = S(i, j);
				point[2] = V(i, j);
				resultHSV(i, j) = point;

			}
		}
		cvtColor(resultHSV, rgbResult, COLOR_HSV2BGR_FULL);
		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);
		imshow("HSV", resultHSV);
		imshow("normal", rgbResult);
		waitKey();
	}
}

boolean isInside(Mat src, int i, int j) {
	if (i < src.rows && i>=0 && j < src.cols && j>=0) {
		return 1;
	}
	else {
		return 0;
	}
}
void testIsInside() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_ <uchar> src = imread(fname, 0);
		int i;
		int j;
		scanf("%d", &i);
		scanf("%d", &j);
		printf("%d",isInside(src, i, j));
	}
}
//LABORATOR 02.03
int* computeHistogram(Mat_<uchar> src) {
		int height = src.rows;
		int width = src.cols;
		int* histoValue= (int*)calloc(256,sizeof(int));
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				histoValue[src(i, j)]++;
			}
		}
		return histoValue;
}
float* computePDF(Mat_<uchar>src) {
		float* histoNorm=(float*)calloc(256,sizeof(float));
		int* histoValues;
		histoValues = computeHistogram(src);
		int N = src.rows * src.cols;
		for (int i = 0; i < 256; i++) {
			histoNorm[i] = ((float)histoValues[i]) / N;
		}
		return histoNorm;
}
int* computeHistoBins(Mat_<uchar> src, int nrOfBins) {
	int height = src.rows;
	int width = src.cols;
	int* histoValues = (int*)calloc(nrOfBins, sizeof(int));
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			histoValues[src(i, j) * nrOfBins / 256]++;
		}
	}
	return histoValues;

}
void testDisplayHistogram() {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
	int* histoValues = computeHistoBins(src,64);
	showHistogram("HistoLab", histoValues, 64, 300);
	free(histoValues);
	waitKey();
}

std::vector<int> findLocalMaxima(Mat_<uchar> src) {
	float* pdf = computePDF(src);
	int WH = 5;
	float TH = 0.0003;
	float v = 0;
	std::vector<int> histoMaxima;
	histoMaxima.push_back(0);
	for (int k = WH; k <= 255 - WH; k++) {
		bool isTrue = true;
		v = 0;
		for (int i = k - WH; i <= k + WH; i++) {
			v = v + pdf[i];
			if (pdf[k] < pdf[i]) {
				isTrue = false;
			}
		}
		v = v / (2 * WH + 1);
		if (pdf[k] > v + TH && isTrue) {
			histoMaxima.push_back(k);
			printf("%d ", k);
		}
	}
	histoMaxima.push_back(255);
	return histoMaxima;
}
int getClosestMax(int x, std::vector<int>localMaxima) {
	int diff = MAXINT;
	int closestMax = MAXINT;
	for (int i:localMaxima) {
		if (abs(i - x) < diff) {
			diff = abs(i - x);
			closestMax = i;
		}
	}
	return closestMax;
}
void multiLevelThreshold() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		std::vector<int> localMaxima = findLocalMaxima(src);
		int height = src.rows;
		int width = src.cols;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				src(i, j) = getClosestMax(src(i, j), localMaxima);
			}
		}
		imshow("result", src);
		waitKey();
	}
}
int checkInterval(int x) {
	if (x >= 0 && x <= 255) {
		return x;
	}
	if (x < 0) {
		return 0;
	}
	if (x > 255) {
		return 255;
	}
}
float checkIntervalFloat(float x) {
	if (x >= 0 && x <= 255) {
		return x;
	}
	if (x < 0) {
		return 0;
	}
	if (x > 255) {
		return 255;
	}
}
void FloydSteinberg() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		std::vector<int> localMaxima = findLocalMaxima(src);
		int height = src.rows;
		int width = src.cols;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int oldPixel = src(i, j);
				int newPixel = getClosestMax(oldPixel, localMaxima);
				src(i, j) = newPixel;
				int error = oldPixel - newPixel;
				if (isInside(src, i, j+1)) {
					src(i, j + 1) = checkInterval(src(i, j + 1) + 7 * error / 16);
				}
				if (isInside(src, i+1, j-1)) {
					src(i+1, j - 1) = checkInterval(src(i+1, j - 1) + 3 * error / 16);
				}
				if (isInside(src, i+1, j)) {
					src(i+1, j) = checkInterval(src(i+1, j) + 5 * error / 16);
				}
				if (isInside(src, i+1, j+1)) {
					src(i+1, j + 1) = checkInterval(src(i+1, j + 1) + error / 16);
				}
			}
		}
		imshow("result", src);
		waitKey();
	}
}

//LABORATOR 09.03
int calculateArea(int showResult, Mat_<uchar> src) {

	int height = src.rows;
	int width = src.cols;
	int area = 0;
	Mat_<Vec3b> final(height, width);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src(i, j) == 0) {
				area++;
				final(i, j)[0] = 0;
				final(i, j)[1] = 255;
				final(i, j)[2] = 0;
			}
			else {
				final(i, j)[0] = 255;
				final(i, j)[1] = 255;
				final(i, j)[2] = 255;
			}
		}
	}
	if (showResult) {
		imshow("source", src);
		imshow("result", final);
		printf("%d", area);
		waitKey();
	}
	return area;
}
void showArea() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		calculateArea(1, src);
	}
}

void calculateEverything(Mat_<uchar> src) {

		int height = src.rows;
		int width = src.cols;
		imshow("original", src);
		int sumR = 0;
		int sumC = 0;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src(i, j) == 0) {
					sumR += i;
					sumC += j;
				}
			}
		}
		int area = calculateArea(0, src);
		sumR = sumR / area;
		sumC = sumC / area;
		Mat_<uchar> src2(height, width); 
		src2 = src.clone();
		line(src2, Point(sumC, sumR - 10), Point(sumC, sumR + 10), Scalar(255, 0, 0), 5,LINE_4);
		line(src2, Point(sumC-10, sumR), Point(sumC+10, sumR), Scalar(255, 0, 0), 5, LINE_4);
		/*for (int i = sumR - 20; i < sumR + 20; i++) {
			src(i, sumC) = 255;
		}
		for (int j = sumC - 20; j < sumC + 20; j++) {
			src(sumR, j) = 255;
		}*/
		imshow("centerOfMass", src2);
		int sum1 = 0;
		int sum2 = 0;
		int sum3 = 0;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src(i, j) == 0) {
					sum1 += (i - sumR) * (j - sumC);
					sum2 += (j - sumC) * (j - sumC);
					sum3 += (i - sumR) * (i - sumR);
				}
			}
		}
		sum1 *= 2;
		double fi = atan2(sum1, sum2 - sum3)/2;
		Mat_<uchar> src3(height, width);
		src3 = src.clone();
		line(src3, Point(sumC, sumR), Point(sumC + cos(fi) * 30, sumR + sin(fi)*30), Scalar(255, 0, 0), 5, LINE_4);
		imshow("tangentOfFi",src3);
		//
		int perimeter = 0;
		Mat_<Vec3b> src4(height, width);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if ((isInside(src, i - 1, j) && src(i - 1, j) != src(i,j)) || (isInside(src, i + 1, j) && src(i + 1, j) != src(i, j)) || (isInside(src, i, j - 1) && src(i, j - 1) != src(i, j)) || (isInside(src, i, j + 1) && src(i, j + 1) != src(i, j))) {
					perimeter++;
					src4(i, j)[0] = 255;
					src4(i, j)[1] = 0;
					src4(i, j)[2] = 0;
				}
				else {
					if (src(i, j) == 0) {
						src4(i, j)[0] = 0;
						src4(i, j)[1] = 0;
						src4(i, j)[2] = 0;
					}
				}
			}
		}
		imshow("perimeter", src4);
		//
		float thinnes = 4 * PI * area / (perimeter * perimeter);
		//
		int maxC = 0;
		int maxR = 0;
		int minC = width;
		int minR = height;
		Mat_<Vec3b> src5(height, width);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src(i, j) == 0) {
					if (i < minR) {
						minR = i;
					}
					if (i > maxR) {
						maxR = i;
					}
					if (j < minC) {
						minC = j;
					}
					if (j > maxC) {
						maxC = j;
					}
					src5(i, j)[0] = 0;
					src5(i, j)[1] = 0;
					src5(i, j)[2] = 0;
				}
				else {
					src5(i, j)[0] = 255;
					src5(i, j)[1] = 255;
					src5(i, j)[2] = 255;
				}
			}

		}
		rectangle(src5, Point(minC, minR), Point(maxC, maxR), Vec3b(255, 0, 0), 3);
		imshow("aspect ratio", src5);
		//
		int* proj = (int*)calloc(height,sizeof(int));
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src(i, j) == 0) {
					proj[i]++;
				}
			}
		}
		Mat_<uchar>src6(height, width);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < proj[i]; j++) {
				src6(i, j) = 0;
			}
		}
		imshow("projection left", src6);
		//
		int* proj2 = (int*)calloc(width, sizeof(int));
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src(i, j) == 0) {
					proj2[j]++;
				}
			}
		}
		Mat_<uchar>src7(height, width);
		for (int j = 0; j < width; j++) {
			for (int i = 0; i < proj2[j]; i++) {
				src7(i,j) = 0;
			}
		}
		imshow("projection up", src7);
		//

		waitKey();
}

void onMouse(int event, int x, int y, int flags, void* param) {
	Mat_<Vec3b>* src = (Mat_<Vec3b>*)param;
	Mat_<uchar> dst = Mat_<uchar>(src->rows, src->cols);
	dst.setTo(255);

	if (event == EVENT_LBUTTONUP)
	{
		for (int i = 0; i < src->rows; i++)
			for (int j = 0; j < src->cols; j++)
				if ((*src)(i, j) == (*src)(y, x))
					dst(i, j) = 0;

		imshow("binary image", dst);
		calculateEverything(dst);
		waitKey();
	}
}
void click() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<Vec3b> src = imread(fname, IMREAD_COLOR);
		namedWindow("New Window", 1);
		setMouseCallback("New Window", onMouse, &src);
		imshow("New Window", src);
		waitKey();
	}
}

//LABORATOR 16.03 
Mat_<int> labelingBFS(Mat_<uchar> src) {

	int height = src.rows;
	int width = src.cols;
	int di[] = { -1,-1,0,1,1,1,0,-1 };
	int dj[] = { 0,1,1,1,0,-1,-1,-1 };
	int label = 0;
	Mat_<int> labels(height, width);
	labels.setTo(0);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src(i, j) == 0 && labels(i, j) == 0) {
				label++;
				labels(i, j) = label;
				std::queue<Point> queueNew;
				queueNew.push({ i,j });
				while (!queueNew.empty()) {
					Point p = queueNew.front();
					queueNew.pop();
					for (int nI = 0; nI < 8; nI++) {
						int i2 = p.x + di[nI];
						int j2 = p.y + dj[nI];
						if (isInside(src,i2,j2)) {
							if (src(i2, j2) == 0 && labels(i2, j2) == 0) {
								labels(i2, j2) = label;
								queueNew.push({ i2, j2 });
							}
						}
					}
				}
			}
		}
	}
	return labels;
}

Mat_<Vec3b> colourLabels(Mat_<int> src) {
	default_random_engine gen;
	uniform_int_distribution<int> d(0, 255);

	int max = 0;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (max < src(i, j)) {
				max = src(i, j);
			}
		}
	}
	Vec3b* colours = (Vec3b*)calloc(max+1, sizeof(Vec3b));
	colours[0][0] = 255;
	colours[0][1] = 255;
	colours[0][2] = 255;
	for (int i = 1; i <= max; i++) {
		uchar x = d(gen);
		colours[i][0] = x;
		x = d(gen);
		colours[i][1] = x;
		x = d(gen);
		colours[i][2] = x;
	}
	Mat_<Vec3b> dst(src.rows, src.cols);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			dst(i, j) = colours[src(i, j)];
		}
	}
	return dst;
}

void testingLabelBFS() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat_<int> labels = labelingBFS(src);

		Mat_<Vec3b> dst = colourLabels(labels);
		imshow("Original", src);
		cout << labels;
		imshow("LabeledColoured", dst);
		waitKey();
	}
}

Mat_<int> labelingTwoPass(Mat_<uchar> src) {
	int height = src.rows;
	int width = src.cols;
	int label = 0;
	Mat_<int> labels(height, width);
	labels.setTo(0);
	vector<vector<int>> edges(10000);
	int di[] = { -1,-1,-1,0,0,1,1,1 };
	int dj[] = { 0,1,-1,-1,1,0,1,-1 };
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src(i, j) == 0 && labels(i, j) == 0) {
				std::vector<int> L;
				Point p = { i,j };
				for (int nI = 0; nI < 4; nI++) {
					int i2 = p.x + di[nI];
					int j2 = p.y + dj[nI];
					if (isInside(src, i2, j2)) {
						if (labels(i2, j2) > 0) {
							L.push_back(labels(i2, j2));
						}
					}
				}
				if (L.size() == 0) {
					label++;
					labels(i, j) = label;
				}
				else {
					int x = *min_element(L.begin(), L.end());
					labels(i, j) = x;
					for (int y : L) {
						if (y != x) {
							edges[x].push_back(y);
							edges[y].push_back(x);
						}
					}
				}

			}
		}
	}
	int newLabel = 0;
	int* newLabels = (int*)calloc(label + 1, sizeof(int));
	for (int i = 1; i <= label; i++) {
		if (newLabels[i] == 0) {
			newLabel++;
			std::queue<int> queueNew;
			newLabels[i] = newLabel;
			queueNew.push(i);
			while (!queueNew.empty()) {
				int x = queueNew.front();
				queueNew.pop();
				for (int y : edges[x]) {
					if (newLabels[y] == 0) {
						newLabels[y] = newLabel;
						queueNew.push(y);
					}
				}
			}
		}
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			labels(i, j) = newLabels[labels(i, j)];
		}
	}
	return labels;
}
void testingLabelTwoPass() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat_<int> labels = labelingTwoPass(src);
		Mat_<Vec3b> dst = colourLabels(labels);
		imshow("Original", src);
		//cout << labels;
		imshow("LabeledColoured", dst);
		waitKey();
	}
}

//LABORATOR 23.03
void borderTracing() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		vector<int> dirs;
		int height = src.rows;
		int width = src.cols;
		int di[8] = { 0,-1,-1,-1,0,1,1,1 };
		int dj[8] = { 1,1,0,-1,-1,-1,0,1 };
		Point2i* pointsB = (Point2i*)malloc(10000 * sizeof(Point2i));
		int noOfPoints = 0;
		bool firstFound = 0;
		Point2i lastPoint;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src(i, j) == 0) {
					pointsB[noOfPoints] = Point2i(i,j);
					lastPoint = Point2i(i,j);
					noOfPoints++;
					firstFound = 1;
					break;
				}
			}
			if (firstFound) {
				break;
			}
		}
		int dirp = 7;
		int dir0;
		while (!(pointsB[0] == pointsB[noOfPoints - 2] && pointsB[1] == pointsB[noOfPoints - 1]) || noOfPoints <= 2) {
			if (dirp % 2 == 0) {
				dir0 = (dirp + 7) % 8;
			}
			else {
				dir0 = (dirp + 6) % 8;
			}
			bool finished = 0;
			for (int k = 0; k < 8 && !finished; k++) {
				int i2 = lastPoint.x + di[(dir0 + k) % 8];
				int j2 = lastPoint.y + dj[(dir0 + k) % 8];
				if (src(i2,j2) == 0) {
					pointsB[noOfPoints] = Point2i(i2,j2);
					noOfPoints++;
					lastPoint = Point2i(i2, j2);
					dir0 = (dir0 + k) % 8;
					dirp = dir0;
					dirs.push_back(dirp);
					finished = 1;
				}
			}
		}
		Mat_<uchar> dst(height, width);
		dst.setTo(0);
		for (int i = 0; i < noOfPoints; i++) {
			dst(pointsB[i].x, pointsB[i].y) = 255;
		}
		imshow("Destination", dst);
		for (int i = 0; i < dirs.size(); i++) {
			cout<<dirs[i]<<" ";
		}
		vector<int> derivate;
		derivate.push_back((dirs[0] - dirs[dirs.size()]+8)%8);
		for (int i = 1; i < dirs.size(); i++) {
			derivate.push_back((dirs[i] - dirs[i - 1] + 8) % 8);
		}
		waitKey();
	}
}

void reconstruct() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> img = imread(fname, IMREAD_GRAYSCALE);
		int height = img.rows;
		int width = img.cols;
		int di[8] = { 0,-1,-1,-1,0,1,1,1 };
		int dj[8] = { 1,1,0,-1,-1,-1,0,1 };
		FILE* f;
		f = fopen("reconstruct.txt", "r");
		int x;
		fscanf(f, "%d", &x);
		int y;
		fscanf(f, "%d", &y);
		int n;
		fscanf(f, "%d", &n);
		vector<int> dirs(n);
		vector<Point2i> pointsB;
		int i2 = x;
		int j2 = y;
		pointsB.push_back(Point2i(x, y));
		for (int i = 0; i < n; i++) {
			fscanf(f, "%d", &dirs[i]);
		}
		for (int i = 0; i < n; i++) {
			i2 = i2 + di[dirs[i]];
			j2 = j2 + dj[dirs[i]];
			pointsB.push_back(Point2i(i2, j2));
		}
		for (int i = 0; i < pointsB.size(); i++) {
			img(pointsB[i].x, pointsB[i].y) = 255;
		}
		imshow("image", img);
		waitKey();
	}

}
void testReadPPm() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<Vec3b> src = imread(fname, IMREAD_COLOR);
		imshow("image", src);
		waitKey();
	}
}

//LABORATOR 30.03
Mat_<uchar> createKernel(int n, bool isCross) {
	Mat_<uchar> kernell(n, n);
	kernell.setTo(0);
	if (isCross) {
		for (int i = 0; i < kernell.rows; i++) {
			kernell(i, kernell.cols / 2) = 0;
		}
		for (int j = 0; j < kernell.cols; j++) {
			kernell(kernell.rows / 2, j) = 0;
		}
		return kernell;
	}
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			kernell(i, j) = 0;
		}
	}
	return kernell;
}

void dillation(Mat_<uchar> kernell) {
	char fname[256];
	while (openFileDlg(fname)) {
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat_<uchar> dst = src.clone();

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src(i, j) == 0) {
					for (int u = 0; u < kernell.rows; u++) {
						for (int v = 0; v < kernell.cols; v++) {
							if (kernell(u, v) == 0 && isInside(src, i + u - kernell.rows / 2, j + v - kernell.cols / 2)) {
								dst(i + u - kernell.rows / 2, j + v - kernell.cols / 2) = 0;
							}
						}
					}
				}
			}
		}
		imshow("Final", dst);
		waitKey();
	}
}

void erosion(Mat_<uchar> kernell) {
	char fname[256];
	while (openFileDlg(fname)) {
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat_<uchar> dst = src.clone();
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src(i, j) == 0) {
					for (int u = 0; u < kernell.rows; u++) {
						for (int v = 0; v < kernell.cols; v++) {
							if (kernell(u, v) == 0 && isInside(src, i + u - kernell.rows / 2, j + v - kernell.cols / 2)) {
								if (src(i + u - kernell.rows / 2, j + v - kernell.cols / 2) != 0) {
									dst(i, j) = 255;
								}
							}
						}
					}
				}
			}
		}
		imshow("Final", dst);
		waitKey();
	}
}

//LABORATOR 06.04

float meanIntensityValue(Mat src) {
	int height = src.rows;
	int width = src.cols;
	int* histoValues = computeHistogram(src);
	int M = width * height;
	float sum = 0;
	for (int i = 0; i < 256; i++) {
		sum += i * histoValues[i];
	}
	sum = sum / M;
	cout <<"Mean Intensity value: " << sum << "\n";
	return sum;
}

float standardDeviation(Mat src) {
	int height = src.rows;
	int width = src.cols;
	float meanIntensity = meanIntensityValue(src);
	float stdDeviation = 0;
	float* normalisedHistoValues = computePDF(src);
	for (int i = 0; i < 256; i++) {
		stdDeviation += (i - meanIntensity) * (i - meanIntensity) * normalisedHistoValues[i];
	}
	stdDeviation = sqrt(stdDeviation);
	cout << "Standard deviation value:" << stdDeviation << "\n";
	return stdDeviation;
}

int* computeCumulativeHistogram(Mat src) {
	int* histoValues = computeHistogram(src);
	int* cumulativeHistoValues = (int*)calloc(256, sizeof(int));
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j <= i; j++) {
			cumulativeHistoValues[i] += histoValues[j];
		}
	}
	return cumulativeHistoValues;
}

void globalThresholding(Mat src) {
	int height = src.rows;
	int width = src.cols;
	int* histoValues = computeHistogram(src);
	int maxI = 0;
	int minI = 0;
	for (int i = 0; i < 256; i++) {
		if (histoValues[i] != 0) {
			minI = i;
			break;
		}
	}
	for (int i = 255; i >= 0; i--) {
		if (histoValues[i] != 0) {
			maxI = i;
			break;
		}
	}
	float thresholdCurr = (minI + maxI) / 2.0;
	float thresholdPred = -1;
	float uG1;
	float uG2;
	int N1;
	int N11;
	int N2;
	int N22;

	while (abs(thresholdCurr - thresholdPred) >= 0.1) {
		N1 = 0;
		N2 = 0;
		N11 = 0;
		N22 = 0;
		uG1 = 0;
		uG2 = 0;
		for (int i = minI; i <= maxI; i++) {
			if (i <= thresholdCurr) {
				N1 += histoValues[i];
				N11 += i * histoValues[i];
			}
			else {
				N2 += histoValues[i];
				N22 += i * histoValues[i];
			}
		}
		uG1 = N11 / N1;
		uG2 = N22 / N2;
		thresholdPred = thresholdCurr;
		thresholdCurr = (uG1 + uG2) / 2.0;
	}
	imshow("Original", src);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src.at<uchar>(i, j) < thresholdCurr) {
				src.at<uchar>(i, j) = 0;
			}
			else {
				src.at<uchar>(i, j) = 255;
			}
		}
	}
	cout << thresholdCurr;
	imshow("Result", src);
	waitKey();
}

int* histogramStretchShrink(Mat src) {
	int height = src.rows;
	int width = src.cols;
	int gOutMin;
	int gOutMax;
	cout << "Enter Minimum value for Stretch/Shrink: ";
	cin >> gOutMin;
	cout << "Enter Maximum value for Stretch/Shrink: ";
	cin >> gOutMax;
	int* histoValues = computeHistogram(src);
	int gInMin;
	int gInMax;
	for (int i = 0; i < 256; i++) {
		if (histoValues[i] != 0) {
			gInMin = i;
			break;
		}
	}
	for (int i = 255; i >= 0; i--) {
		if (histoValues[i] != 0) {
			gInMax = i;
			break;
		}
	}
	imshow("Original", src);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			src.at<uchar>(i, j) = gOutMin + (src.at<uchar>(i, j) - gInMin) * (gOutMax - gOutMin) / (gInMax - gInMin);
		}
	}
	int* histoValuesStretchShrink = (int*)calloc(256, sizeof(int));
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			histoValuesStretchShrink[src.at<uchar>(i, j)]++;
		}
	}
	imshow("Stretch/Shrink", src);
	showHistogram("Stretch/Shrink Histogram",histoValuesStretchShrink, 256, 300);
	waitKey();
	return histoValuesStretchShrink;
}

void gammaCorrection(Mat src) {
	int height = src.rows;
	int width = src.cols;
	float gammaCoeff;
	cout << "Enter a value for gamma coefficient: ";
	cin >> gammaCoeff;
	imshow("Original", src);
	float* gammaCorrected=(float*)calloc(256,sizeof(float));
	for (int i = 0; i < 256; i++) {
		gammaCorrected[i]= checkIntervalFloat(255 * pow((float)i / 255, gammaCoeff));
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			src.at<uchar>(i, j) = gammaCorrected[src.at<uchar>(i, j)];
		}
	}
	imshow("Gamma Correction", src);
	waitKey();
}

void histogramEqualization(Mat src) {
	int* histoValues = computeCumulativeHistogram(src);
	int height = src.rows;
	int width = src.cols;
	int M = height * width;
	int* histoValuesFinal = (int*)calloc(256, sizeof(int));
	int* histoValuesEqualized = (int*)calloc(256, sizeof(int));
	for (int i = 0; i < 256; i++) {
		histoValuesFinal[i] = 255 * histoValues[i]/M;
	}
	imshow("Original", src);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			src.at<uchar>(i, j) = histoValuesFinal[src.at<uchar>(i, j)];
			histoValuesEqualized[src.at<uchar>(i, j)]++;
		}
	}
	imshow("Equalized", src);
	showHistogram("Equalized Histogram", histoValuesEqualized, 256, 300);
}
void testAll() {
	char fname[256];
	while (openFileDlg(fname)) {
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		//standardDeviation(src);
		//globalThresholding(src);
		histogramStretchShrink(src);
		//gammaCorrection(src);
		//histogramEqualization(src);
		waitKey();
	}
}

//LABORATOR 13.04
Mat_<float> createKernelMean() {
	Mat_<float> kernel(3, 3);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			kernel(i, j) = 1;
		}
	}
	return kernel;
}

Mat_<float> createKernelGaussian() {
	Mat_<float> kernel(3, 3);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			kernel(i, j) = 1;
		}
	}
	return kernel;
}

Mat_<float> createKernelLaplace() {
	Mat_<float> kernel(3, 3);
	kernel(0, 0) = 0;
	kernel(0, 1) = -1;
	kernel(0, 2) = 0;
	kernel(1, 0) = -1;
	kernel(1, 1) = 4;
	kernel(1, 2) = -1;
	kernel(2, 0) = 0;
	kernel(2, 1) = -1;
	kernel(2, 2) = 0;
	return kernel;
}

Mat_<float> convolution(Mat_<uchar> img, Mat_<float> kernel) {

	int imgHeight = img.rows;
	int imgWidth = img.cols;
	int kernelHeight = kernel.rows;
	int kernelWidh = kernel.cols;
	Mat_<float> dst(imgHeight, imgWidth);
	dst.setTo(0);
	for (int i = 0; i < imgHeight; i++) {
		for (int j = 0; j < imgWidth; j++) {
			for (int u = 0; u < kernelHeight; u++) {
				for (int v = 0; v < kernelWidh; v++) {
					if (isInside(img, i + u - kernelHeight / 2, j + v + kernelWidh / 2)) {
						dst(i,j) += kernel(u, v) * img(i + u - kernelHeight / 2, j + v + kernelWidh / 2);
					}
				}
			}
		}
	}
	return dst;
}

Mat_<uchar> normalization(Mat_<float> imgConv, Mat_<float> kernel, bool highPassMethod) {
	int imgHeight = imgConv.rows;
	int imgWidth = imgConv.cols;
	int kernelHeight = kernel.rows;
	int kernelWidh = kernel.cols;
	float sumPos = 0;
	float sumNeg = 0;
	Mat_<uchar> dst(imgHeight, imgWidth);

	for (int i = 0; i< kernelHeight; i++) {
		for (int j = 0; j < kernelWidh; j++) {
			if (kernel(i, j) > 0) {
				sumPos += kernel(i, j);
			}
			else {
				sumNeg += kernel(i, j);
			}
		}
	}
	if (sumNeg != 0) {
		for (int i = 0; i < imgHeight; i++) {
			for (int j = 0; j < imgWidth; j++) {
				if (highPassMethod == 0) {
					dst(i, j) = abs(imgConv(i, j)) / max(sumPos, -sumNeg);
				}
				else {
					dst(i, j) = 128 + imgConv(i, j) / (2 * max(sumPos, -sumNeg));
				}
			}
		}
		return dst;
	}
	for (int i = 0; i < imgHeight; i++) {
		for (int j = 0; j < imgWidth; j++) {
			dst(i, j) = imgConv(i, j) / sumPos;
		}
	}
	return dst;

}
void testSpatialFilter() {
	
	char fname[256];
	while (openFileDlg(fname)) {
		Mat_<float> kernel = createKernelLaplace();
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		Mat_<float> convolutedSrc = convolution(src, kernel);
		Mat_<uchar> normalisedSrcAbs = normalization(convolutedSrc, kernel, 0);
		//Mat_<uchar> normalisedSrc128 = normalization(convolutedSrc, kernel, 1);
		imshow("Original", src);
		imshow("Final 1", normalisedSrcAbs);
		//imshow("Final 2", normalisedSrc128);
		waitKey();
	}
}

int main()
{
	//testNegativeImage();
	//testAdditiveFactor(150);
	//testMultiplicativeFactor(0.1);
	//testImageOpenAndSave();
	//testSquaresImage();
	//testInverse();
	//image2RGB2Grey();
	//testGray2BW();
	//testRGB2HSV();
	//testIsInside();
	//testDisplayHistogram();
	//multiLevelThreshold();
	//FloydSteinberg();
	//showArea();
	//calculateCenterOfMass();
	//click();
	//testingLabelBFS();
	//testingLabelTwoPass();
	//borderTracing();
	//reconstruct();
	//testReadPPm();
	//Mat_<uchar> kernellB = createKernel(6, true);
	//dillation(kernellB);
	//erosion(kernellB);
	//testAll();
	testSpatialFilter();
	return 0;
}