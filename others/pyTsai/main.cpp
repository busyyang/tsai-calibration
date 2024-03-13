#include "tsai/cal_main.h"
#include <stdio.h>

#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>


void show_errors(std::string window_name)//, calibration_data tsai_cd, camera_parameters tsai_cp)
{
	double Xf, Yf;
	cv::Mat preview = cv::Mat::zeros(tsai_cp.Ncy, tsai_cp.Ncx, CV_8UC3);
	for (int i = 0; i < tsai_cd.point_count; i++)
	{
		/* calculate the ideal location of the image of the data point */
		world_coord_to_image_coord(tsai_cd.xw[i], tsai_cd.yw[i], tsai_cd.zw[i], &Xf, &Yf);
		cv::circle(preview, cv::Point(Xf, Yf), 10, cv::Scalar(0, 0, 255), 2);
		cv::circle(preview, cv::Point(tsai_cd.Xf[i], tsai_cd.Yf[i]), 10, cv::Scalar(255, 0), 2);
	}
	cv::imshow(window_name, preview);
	cv::waitKey(0);
}


void set_data()
{
	//set data
	std::vector<std::vector<double>> data = {
#include "points.data"
	};
	tsai_cd.point_count = data.size();
	for (size_t i = 0; i < data.size(); i++)
	{
		tsai_cd.xw[i] = data[i][0];
		tsai_cd.yw[i] = data[i][1];
		tsai_cd.zw[i] = data[i][2];
		tsai_cd.Xf[i] = data[i][3];
		tsai_cd.Yf[i] = data[i][4];
	}
}

void set_inner_data()
{
	//set data
	std::vector<std::vector<double>> data = {
#include "points_inner.data"
	};
	tsai_cd.point_count = data.size();
	for (size_t i = 0; i < data.size(); i++)
	{
		tsai_cd.xw[i] = data[i][0];
		tsai_cd.yw[i] = data[i][1];
		tsai_cd.zw[i] = data[i][2];
		tsai_cd.Xf[i] = data[i][3];
		tsai_cd.Yf[i] = data[i][4];
	}
}


static cv::Mat angle2mat(const double& Rx, const double& Ry, const double& Rz)
{
	double sa = sin(Rx);
	double ca = cos(Rx);
	double sb = sin(Ry);
	double cb = cos(Ry);
	double sg = sin(Rz);
	double cg = cos(Rz);

	cv::Mat rm(3, 3, CV_64FC1);

	rm.at<double>(0, 0) = cb * cg;
	rm.at<double>(0, 1) = cg * sa * sb - ca * sg;
	rm.at<double>(0, 2) = sa * sg + ca * cg * sb;
	rm.at<double>(1, 0) = cb * sg;
	rm.at<double>(1, 1) = sa * sb * sg + ca * cg;
	rm.at<double>(1, 2) = ca * sb * sg - cg * sa;
	rm.at<double>(2, 0) = -sb;
	rm.at<double>(2, 1) = cb * sa;
	rm.at<double>(2, 2) = ca * cb;

	return rm;
}


int main_pyTsai()
{

	double image_w = 1024;
	double image_h = 1024;
	// 内参
	double f = 1011;
	double dx = 0.209, dy = 0.209;
	double u0 = 505, v0 = 509.1;

	// 外参
	double rx = 0.001;
	double ry = -0.001;
	double rz = 0.002;

	double tx = 10;
	double ty = -10;
	double tz = 950;

	// 畸变参数
	double k1 = 1;
	double k2 = 100;
	double p1 = -0.01;
	double p2 = -0.01;
	double s1 = -0.02;
	double s3 = 0.01;


	cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << f / dx, 0, u0, 0, f / dy, v0, 0, 0, 1);
	cv::Mat camera_matrix_undistortion = (cv::Mat_<double>(3, 3) << f / dx, 0, (image_w - 1.0) / 2, 0, f / dy, (image_h - 1.0) / 2, 0, 0, 1);
	cv::Mat rotation = angle2mat(rx, ry, rz);
	cv::Mat rvec = (cv::Mat_<double>(3, 1) << rz, ry, rz);
	cv::Mat translation = (cv::Mat_<double>(3, 1) << tx, ty, tz);

	// k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4
	cv::Mat distortion_coeff = (cv::Mat_<double>(12, 1) << k1, k2, p1, p2, 0, 0, 0, 0, s1, 0, s3, 0);

	std::vector<cv::Vec3d> points3D =
	{
#include "marker3d.data"
	};

	std::vector<cv::Vec2d> points2D;
	cv::projectPoints(points3D, rvec, translation, camera_matrix, distortion_coeff, points2D);

	tsai_cd.point_count = points3D.size();
	for (size_t i = 0; i < tsai_cd.point_count; i++)
	{
		tsai_cd.xw[i] = points3D[i][0];
		tsai_cd.yw[i] = points3D[i][1];
		tsai_cd.zw[i] = points3D[i][2];
		tsai_cd.Xf[i] = points2D[i][0];
		tsai_cd.Yf[i] = points2D[i][1];
	}






	// Initial camera parameters
	tsai_cp.Ncx = 1024;
	tsai_cp.Ncy = 1024;
	tsai_cp.dx = 0.209;
	tsai_cp.dy = 0.209;
	tsai_cp.Cx = (tsai_cp.Ncx - 1.0) / 2.0;
	tsai_cp.Cy = (tsai_cp.Ncy - 1.0) / 2.0;
	tsai_cp.sx = 1.0;

	//set_data();

	/*
	if (!noncoplanar_calibration_with_full_optimization())
	{
		printf("Errors when Tsai Calibration! Please have a check!\n");
	}*/


	/* start with a 3 parameter (Tz, f, kappa1) optimization */
	if (!ncc_three_parm_optimization())
		return 0;
	printf("==============Only Tz,f,k1 optimization==============\n");
	print_parameters(&tsai_cp);
	print_constants(&tsai_cc);
	print_errors();
	show_errors("Only Tz,f,k1 optimization");

	//set_data();

	/* do a full optimization minus the image center */
	if (!ncc_nic_optimization())
		return 0;

	printf("==============optimization minus the image center==============\n");
	print_parameters(&tsai_cp);
	print_constants(&tsai_cc);
	print_errors();
	show_errors("optimization minus the image center");

	/* do a full optimization including the image center */
	if (!ncc_full_optimization())
		return 0;

	printf("==============full optimization==============\n");
	print_parameters(&tsai_cp);
	print_constants(&tsai_cc);
	print_errors();
	show_errors("full optimization");

	//print_parameters(&tsai_cp);
	//print_constants(&tsai_cc);
	//print_errors();
	

	return 0;
}