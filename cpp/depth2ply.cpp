#include<opencv2\opencv.hpp>
#include<iostream>
#include<string>
#include<fstream>
#include<vector>
#include<sstream>
#include<math.h>
#include<random>
#include"realsense.h"


// convert depth image to ply
int main(int argc, char* argv[]) {

// file Names
std::string input_color = { argv[1] };
std::string input_depth = { argv[2] };
std::string output_ply = { argv[3] };


// raw images
cv::Mat depthImage = cv::imread(input_depth, -1);
cv::Mat colorImage = cv::imread(input_color, 1);
//
int height = depthImage.rows;
int width = depthImage.cols;

// camera parameter
std::vector<double> dD = { 0.18322406709194183, -0.05855391174554825, 0.005057608708739281, 0.0016838484443724155, 0.2501026391983032 };
std::vector<double> dK = { 476.6209411621094, 0.0, 310.7926025390625, 0.0, 476.6208190917969, 246.08285522460938, 0.0, 0.0, 1.0 };
std::vector<double> cT = { 0.0257,0.00015021,0.00393217 };
std::vector<double> cRQ = { -2.11274768e-03,-4.89503966e-04,-2.14389055e-03,9.99995351e-01 };
std::vector<std::vector<double>> cRM = { { 9.99990328e-01,-4.28569277e-03,9.88062380e-04 },
{ 4.28982957e-03,9.99981880e-01,-4.22337683e-03 },
{ -9.69944380e-04,4.22757460e-03,9.99990593e-01 } };
std::vector<double> cK = { 620.9033813476562, 0.0, 320.0282287597656, 0.0, 620.9034423828125, 242.6624755859375, 0.0, 0.0, 1.0 };
std::vector<double> cD = { 0.0, 0.0, 0.0, 0.0, 0.0 };

convertPlyWithColor(img2depth(depthImage), colorImage, cK, cD, output_ply);

return 0;
}

