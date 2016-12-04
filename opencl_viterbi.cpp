// opencl_test1.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


#include <stdio.h>
#include <CL/cl.h>
#include "CImg.h"
#include <time.h>
#include <memory>
#include <string>
#include "Viterbi.h"

using namespace cimg_library;
using namespace VITERBI;

//image settings
const char IMG_FILE[] = "line_7.bmp";
const int G_LOW = -2;
const int G_HIGH = 2;

void rgb2Gray(const CImg<unsigned char> &img, CImg<unsigned char> &grayImg)
{
	cimg_forXY(img, x, y)
	{
		int R = (int)img(x, y, 0, 0);
		int G = (int)img(x, y, 0, 1);
		int B = (int)img(x, y, 0, 2);

		// obtain gray value from different weights of RGB channels
		//int gray_value = (int)(0.299 * R + 0.587 * G + 0.114 * B);
		int gray_value = (int)((R + G + B) / 3);
		grayImg(x, y, 0, 0) = gray_value;
	}
}

int initializeCL(cl_command_queue &command_queue, cl_context &context, cl_device_id &device_id)
{
	cl_int err;
	cl_uint numPlatforms;
	cl_platform_id platform_id = NULL;

	err = clGetPlatformIDs(1, &platform_id, &numPlatforms);
	if (CL_SUCCESS == err)
	{
		printf("\nDetected OpenCl platforms :%d", numPlatforms);
	}
	else
	{
		printf("\nError calling get platforms. error code :%d", err);
		getchar();
		return 0;
	}

	//selecting device id - default - gpu
	device_id = NULL;
	cl_uint numOfDevices = 0;

	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &numOfDevices);
	if (CL_SUCCESS == err)
	{
		printf("\nSelected device id :%d", device_id);
	}
	else
	{
		printf("\nError calling getdeviceIds. error code :%d", err);
		return err;
	}

	//creating context
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
	if (CL_SUCCESS == err)
	{
		printf("\nCreated context");
	}
	else
	{
		printf("\nError creating context. error code :%d", err);
		return err;
	}

	//creating command queue 
	command_queue = clCreateCommandQueue(context, device_id, 0, &err);
	if (CL_SUCCESS == err)
	{
		printf("\nCreated command queue");
	}
	else
	{
		printf("\nError creating command queue. error code :%d", err);
		return err;
	}
	return CL_SUCCESS;
}

int main(void)
{
	cl_device_id device_id = NULL;
	cl_command_queue command_queue = NULL;
	cl_context context = NULL;
	//initalize OpenCL 
	if (CL_SUCCESS != initializeCL(command_queue, context, device_id))
	{
		printf("\nFailed OpenCl initialization!\n");
		return 0;
	}
	CImg<unsigned char> img(IMG_FILE);
	CImg<unsigned char> gray_img(img.width(), img.height(), 1, 1, 0);
	//convert to monochrome image
	rgb2Gray(img, gray_img);
	
	std::unique_ptr<unsigned char> img_out(new unsigned char[img.width() * img.height()]);
	memcpy(img_out.get(), gray_img.data(0, 0, 0, 0), img.width() * img.height());
	
	std::unique_ptr<int> line_x(new int[img.width()]);
	
	clock_t start = clock();
	viterbiLineOpenCL_cols(img_out.get(), img.height(), img.width(), line_x.get(), G_LOW, G_HIGH, command_queue, context, device_id);
	clock_t end = clock();
	double time_ms = (double)(end - start);
	for (int i = 0; i < img.width(); i++)
	{
		img(i, (int)line_x.get()[i], 0, 0) = 255;
		img(i, (int)line_x.get()[i], 0, 1) = 0;
		img(i, (int)line_x.get()[i], 0, 2) = 0;
	}
	CImgDisplay gray_disp(gray_img, "image gray");
	CImgDisplay rgb_disp(img, "image rgb1");
	printf("\nviterbi parallel time %f ms\n", time_ms);
	
	start = clock();
	viterbiLineDetect(img_out.get(), (unsigned int)img.height(), (unsigned int)img.width(), (unsigned int*)line_x.get(), G_LOW, G_HIGH);
	end = clock();
	time_ms = (double)(end - start);
	for (int i = 0; i < img.width(); i++)
	{
		img(i, (int)line_x.get()[i], 0, 0) = 255;
		img(i, (int)line_x.get()[i], 0, 1) = 0;
		img(i, (int)line_x.get()[i], 0, 2) = 0;
	}
	printf("\nViterbi serial time %f ms\n", time_ms);
	CImgDisplay rgb1_disp(img, "Image rgb2");
	while (!rgb1_disp.is_closed());
	//cleanup
	int err = 0;
	err = clFlush(command_queue);
	err = clFinish(command_queue);
	err = clReleaseCommandQueue(command_queue);
	err = clReleaseContext(context);
	return 0;
}

