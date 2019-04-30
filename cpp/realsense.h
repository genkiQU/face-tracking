#pragma once
#include<opencv2\opencv.hpp>
#include<iostream>
#include<string>
#include<fstream>
#include<vector>
#include<sstream>
#include<math.h>
#include<random>

#define epsilon 0.0001

cv::Mat img2depth(cv::Mat img) {
	//std::cout << "img -> depth" << std::endl;
	int height = img.rows;
	int width = img.cols;
	//std::cout << "width : " << width << "  height : " << height << std::endl;
	cv::Mat depth = cv::Mat::zeros(height, width, CV_32F);

	for (int h = 0; h < height; h++) {
		cv::Vec4b *src = img.ptr<cv::Vec4b>(h);
		float *depthPtr = depth.ptr<float>(h);
		for (int w = 0; w < width; w++) {
			float val = float(int(src[w][1])) * 256.0*256.0 + float(int(src[w][2])) * 256.0 + float(int(src[w][3]));
			float f = ldexp(1.0, int(src[w][0]) - (128 + 24));
			depthPtr[w] = f*(val + 0.5f);
			//depth.at<float>(h, w) = f*(val + 0.5f);
		}
	}
	return depth;
}

cv::Mat depth2img(cv::Mat depth) {
	//std::cout << "depth -> img" << std::endl;
	int height = depth.rows;
	int width = depth.cols;
	//std::cout << "width : " << width << "  height : " << height << std::endl;
	cv::Mat img = cv::Mat::zeros(cv::Size(width, height), 24);// 4 channel image

	for (int h = 0; h < height; h++) {
		cv::Vec4b *src = img.ptr<cv::Vec4b>(h);
		float *depthPtr = depth.ptr<float>(h);
		for (int w = 0; w < width; w++) {
			float depth = depthPtr[w];
			int e = -128;
			long val = 0;
			if (fabs(depth) > 1e-32) {
				val = frexp(depth, &e)*256.0f*256.0f*256.0f;
			}
			src[w][0] = (uchar)(e + 128);
			src[w][1] = (uchar)((val & 0x00ff0000) >> 16);
			src[w][2] = (uchar)((val & 0x0000ff00) >> 8);
			src[w][3] = (uchar)((val & 0x000000ff));
		}
	}

	return img;
}

std::vector<double> deprojection_pixel_to_point(std::vector<double> K, std::vector<double> D, std::vector<double> pixel, double depth) {
	std::vector<double> point = { 0.0,0.0,0.0 };

	double x = (pixel[0] - K[2]) / K[0];
	double y = (pixel[1] - K[5]) / K[4];
	//std::cout << x << " " << y << std::endl;
	double r2 = x*x + y*y;

	double f = 1.0 + 2.0*D[0] * r2 + D[1] * r2*r2 + D[4] * r2*r2*r2;
	double ux = x*f + 2.0 * D[2] * x*y + D[3] * (r2 + 2.0 * x*x);
	double uy = y*f + 2.0 * D[3] * x*y + D[2] * (r2 + 2.0 * y*y);

	point[0] = ux*depth;
	point[1] = uy*depth;
	point[2] = depth;

	return point;
}

std::vector<double> transform_point_to_point(std::vector<double> from_point, std::vector<double>T, std::vector<std::vector<double>>R) {
	std::vector<double> to_point = { 0.0,0.0,0.0 };

	to_point[0] = R[0][0] * from_point[0] + R[0][1] * from_point[1] + R[0][2] * from_point[2] + T[0];
	to_point[1] = R[1][0] * from_point[0] + R[1][1] * from_point[1] + R[1][2] * from_point[2] + T[1];
	to_point[2] = R[2][0] * from_point[0] + R[2][1] * from_point[1] + R[2][2] * from_point[2] + T[2];

	return to_point;
}

std::vector<double> project_point_to_pixel(std::vector<double> point, std::vector<double> K) {
	std::vector<double> pixel = { 0.0,0.0 };
	double x = point[0] / point[2];
	double y = point[1] / point[2];

	pixel[0] = x*K[0] + K[2];
	pixel[1] = y*K[4] + K[5];

	return pixel;
}

cv::Mat depthComplesion(cv::Mat depthData) {
	int height, width;
	width = depthData.cols;
	height = depthData.rows;
	//cv::Mat maskImage = cv::Mat::zeros(height, width, CV_8U);

	for (int h = 1; h < height-1; h++) {
		float *depthPtr = depthData.ptr<float>(h);
		for (int w = 1; w < width-1; w++) {
			float depth = depthPtr[w];
			if (depth >= epsilon)continue;

			float validSum = 0.0;
			float validNum = 0.0;

			for (int i = 0; i < 3; i++) {
				float *maskDepthPtr = depthData.ptr<float>(h + (i - 1));
				for (int j = 0; j < 3; j++) {
					float maskDepth = maskDepthPtr[w + (j - 1)];
					if (maskDepth < epsilon)continue;
					validNum += 1.0;
					validSum += maskDepth;
				}
			}
			if (validSum < 2.0)continue;
			depthPtr[w] = validSum / validNum;
		}
	}
	return depthData;
}

std::vector<std::string> split(const std::string &s, char delim) {
	std::vector<std::string> elems;
	std::stringstream ss(s);
	std::string item;

	while (std::getline(ss, item, delim)) {
		if (!item.empty()) {
			elems.push_back(item);
		}
	}
	return elems;
}

cv::Mat align(cv::Mat depthData, int width, int height, std::vector<double> dK, std::vector<double> dD,
	std::vector<double>cT, std::vector<std::vector<double>> cRM, std::vector<double>cK) {
	cv::Mat depthAlignData = cv::Mat::zeros(height, width, CV_32F);
	cv::Mat depthMemoryMap = cv::Mat::zeros(height, width, CV_32F);
	for (int h = 0; h < height; h++) {
		float *depthPtr = depthData.ptr<float>(h);
		for (int w = 0; w < width; w++) {
			double depth = double(depthPtr[w]);
			if (depth < epsilon)continue;
			std::vector<double> depthPixel = { 0.0,0.0 };
			depthPixel[0] = double(w) - 0.5;
			depthPixel[1] = double(h) - 0.5;
			//std::cout << "deproj" << std::endl;
			//std::cout << depthPixel[0] << " " << depthPixel[1] << " " << depth << std::endl;
			std::vector<double> depthPoint = { 0.0,0.0,0.0 };
			std::vector<double> otherPoint = { 0.0,0.0,0.0 };
			std::vector<double> otherPixel = { 0.0,0.0 };
			//std::cout << "deproj" << std::endl;
			depthPoint = deprojection_pixel_to_point(dK, dD, depthPixel, depth);
			//std::cout << "deproj" << std::endl;
			otherPoint = transform_point_to_point(depthPoint, cT, cRM);
			//std::cout << "deproj" << std::endl;
			otherPixel = project_point_to_pixel(otherPoint, cK);
			//std::cout << "deproj" << std::endl;

			int other_x0 = int(otherPixel[0] + 0.5);
			int other_y0 = int(otherPixel[1] + 0.5);

			if (other_x0 < 0.0 || other_y0 < 0.0)continue;

			depthPixel[0] = double(w) + 0.5;
			depthPixel[1] = double(h) + 0.5;

			depthPoint = deprojection_pixel_to_point(dK, dD, depthPixel, depth);
			otherPoint = transform_point_to_point(depthPoint, cT, cRM);
			otherPixel = project_point_to_pixel(otherPoint, cK);

			int other_x1 = int(otherPixel[0] + 0.5);
			int other_y1 = int(otherPixel[1] + 0.5);

			if (other_x1 >= width || other_y1 >= height)continue;
			//std::cout << other_x0 << " " << other_y0 << " " << other_x1 << " " << other_y1 << std::endl;
			for (int y = other_y0; y <= other_y1; y++) {
				for (int x = other_x0; x <= other_x1; x++) {
					float depthMemory = depthMemoryMap.at<float>(y, x);
					if (depthMemory <0.0001 || depthMemory > depth) {
						depthAlignData.at<float>(y, x) = depthData.at<float>(h, w);
					}
				}
			}

		}
	}
	depthAlignData = depthComplesion(depthAlignData);
	return depthAlignData;
}

void convertPlyWithColor(cv::Mat depthMat, cv::Mat colorMat, std::vector<double> K, std::vector<double>D, std::string plyfile) {

	std::ofstream ofs;
	//std::string tempfile = "temp.txt";
	//ofs.open(tempfile, std::ios::out);

	int height, width;
	width = depthMat.cols;
	height = depthMat.rows;
	int pointNum = 0;
	std::vector<std::string> stringList;


	for (int h = 0; h < height; h++) {
		float *depthPtr = depthMat.ptr<float>(h);
		cv::Vec3b *colorPtr = colorMat.ptr<cv::Vec3b>(h);
		for (int w = 0; w < width; w++) {
			double depth = double(depthPtr[w]);
			if (depth < epsilon)continue;
			cv::Vec3b color = colorPtr[w];
			std::vector<double> pixel = { 0.0,0.0 };
			pixel[0] = double(w);
			pixel[1] = double(h);

			std::vector<double> point = { 0.0,0.0,0.0 };

			point = deprojection_pixel_to_point(K, D, pixel, depth);


			std::string str = "" + std::to_string(point[0]) + " " + std::to_string(point[1]) + " " +
				std::to_string(point[2]) + " " + std::to_string(int(color[2])) + " " + std::to_string(int(color[1])) + " " + std::to_string(int(color[0]));
			stringList.push_back(str);
			//ofs << point[0] << " " << point[1] << " " << point[2] << " " << int(color[2]) << " " << int(color[1])<< " " << int(color[0]) << std::endl;
			pointNum++;
		}
	}
	//ofs.close();

	ofs.open(plyfile, std::ios::out);
	// write to ply file
	ofs << "ply\n";
	ofs << "format ascii 1.0\n";
	ofs << "element vertex " << pointNum << "\n";
	ofs << "property float x\n";
	ofs << "property float y\n";
	ofs << "property float z\n";
	ofs << "property uchar red\n";
	ofs << "property uchar green\n";
	ofs << "property uchar blue\n";
	ofs << "end_header\n";

	for (int i = 0; i < stringList.size(); i++) {
		ofs << stringList[i] << std::endl;
	}
	ofs.close();
}

void reprojectionPC_color(std::vector<std::vector<double>> pointList, std::vector<std::vector<int>> colorList,
	int width, int height, std::vector<double> K, std::string colorFile, std::string depthFile) {
	cv::Mat depthImage = cv::Mat::zeros(cv::Size(width, height), 24);
	cv::Mat depthMemoryMat = cv::Mat::zeros(cv::Size(width, height), CV_32F);
	cv::Mat colorImage = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);

	//cv::Mat depthMat = img2depth(depthImage);

	int pointTotalNum = pointList.size();

	for (int pointNum = 0; pointNum < pointTotalNum; pointNum++) {
		std::vector<double> point = pointList[pointNum];
		std::vector<int> color = colorList[pointNum];
		std::vector<double> pixel = { 0.0,0.0 };
		float depth = float(point[2]);

		pixel = project_point_to_pixel(point, K);
		int x = int(pixel[0] + 0.5);
		int y = int(pixel[1] + 0.5);

		if (x < 0 || y < 0 || x >= width || y >= height)continue;

		float *depthMemoryPtr = depthMemoryMat.ptr<float>(y);
		float depthMemory = depthMemoryPtr[x];
		if (depth > depthMemory && depthMemory > epsilon)continue;

		depthMemoryPtr[x] = depth;

		cv::Vec3b *colorImagePtr = colorImage.ptr<cv::Vec3b>(y);
		for (int i = 0; i < 3; i++) {
			colorImagePtr[x][i] = color[2 - i];
		}
	}
	cv::imwrite(colorFile, colorImage);
	cv::imwrite(depthFile, depth2img(depthMemoryMat));
}