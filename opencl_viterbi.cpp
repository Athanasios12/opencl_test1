#include <stdio.h>
#include <CL/cl.h>
#include "CImg.h"
#include <time.h>
#include <memory>
#include <string>
#include <vector>
#include "Viterbi.h"

using namespace cimg_library;
using namespace VITERBI;

//image settings
const char IMG_FILE[] = "line_4.bmp";
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

//maybe add clr tabel creation for python/matlab representation or gnuplot
typedef struct
{
	uint32_t m_img_num;
	std::string m_img_name;
	uint32_t m_img_size;
	int m_g_low;
	int m_g_high;
	std::vector<double> m_exec_time;
	std::vector<std::vector<unsigned int> > m_lines_pos;
}PlotData;

typedef struct
{
	std::vector<PlotData> m_pData;
	uint32_t m_plot_id;
}PlotInfo;

//maybe later add returning error codes
void test_viterbi(const std::vector<std::string> &img_files, int g_high, int g_low, int g_incr, PlotInfo &plotInfo)
{
	if (g_high <= g_low || g_high < 0 || g_low > 0)
	{
		return;
	}
	cl_device_id device_id = NULL;
	cl_command_queue command_queue = NULL;
	cl_context context = NULL;
	//initalize OpenCL 
	if (CL_SUCCESS != initializeCL(command_queue, context, device_id))
	{
		printf("\nFailed OpenCl initialization!\n");
		return;
	}
	uint32_t img_num = 0;
	for (auto && img_file : img_files)
	{
		CImg<unsigned char> img(img_file.c_str());
		CImg<unsigned char> gray_img(img.width(), img.height(), 1, 1, 0);
		//convert to monochrome image
		rgb2Gray(img, gray_img);
		std::unique_ptr<unsigned char> img_out(new unsigned char[img.width() * img.height()]);
		memcpy(img_out.get(), gray_img.data(0, 0, 0, 0), img.width() * img.height());

		PlotData data;
		data.m_img_name = img_file;
		data.m_img_num = img_num;
		data.m_img_size = img.width() * img.height();

		while (g_high + abs(g_low) > 0 && g_low <= 0 && g_high >= 0) // maybe redundant but whatever...
		{
			std::vector<std::vector<unsigned int> > line_results;
			std::vector<double> times;
			std::vector<unsigned int> line_x(img.width(), 0);
			//viterbi parallel cols gpu version 
			clock_t start = clock();
			viterbiLineOpenCL_cols(img_out.get(), img.height(), img.width(), &line_x[0], g_low, g_high, command_queue, context, device_id);
			clock_t end = clock();
			double time_ms = (double)(end - start);

			line_results.push_back(line_x);
			times.push_back(time_ms);
			//viterbi serial cpu version
			start = clock();
			viterbiLineDetect(img_out.get(), img.height(), img.width(), line_x, g_low, g_high);
			end = clock();
			time_ms = (double)(end - start);

			line_results.push_back(line_x);
			times.push_back(time_ms);

			start = clock();
			//viterbi parallel rows gpu version
			viterbiLineOpenCL_rows(img_out.get(), img.height(), img.width(), &line_x[0], g_low, g_high, command_queue, context, device_id);
			end = clock();
			time_ms = (double)(end - start);

			line_results.push_back(line_x);
			times.push_back(time_ms);

			//... later add cpu opecl version + parallel std thread version +  eventually cpu + gpu combo opencl

			//save data for ploting and tabel representation
			data.m_g_high = g_high;
			data.m_g_low = g_low;
			data.m_exec_time = times;
			data.m_lines_pos = line_results;
			//save plot data in plot info data vector
			plotInfo.m_pData.push_back(data);
			//next g_l/h combo
			g_low += g_incr;
			g_high -= g_incr;
			//here maybe necessery to add line
			//err = clFlush(command_queue); for cleaning command queue, maybe even earlier with next opencl call
			//between viterbi cols and rows function calls
		}
		++img_num;
	}
	//cleanup
	int err = 0;
	err = clFlush(command_queue); // maybe put it inside the loop ?
	err = clFinish(command_queue);
	err = clReleaseCommandQueue(command_queue);
	err = clReleaseContext(context);
}

void generateCsv(const std::string &file, const PlotInfo &pInfo, const std::vector<std::string> &columnNames)
{
	std::ofstream data_file;
	data_file.open(file);
	uint32_t i = 0;
	for (auto && column : columnNames)
	{
		if (i > 0)
		{
			data_file << "," << column;
		}
		else
		{
			data_file << column;
		}
		++i;
	}
	data_file << "\n";
	for (auto && data : pInfo.m_pData)
	{
		data_file << data.m_img_num << "," << data.m_img_size << "," << data.m_g_low << "," << data.m_g_high;
		for (auto && time : data.m_exec_time)
		{
			data_file << "," << time;
		}
		data_file << "\n";
	}
}

int main(void)
{
	//basic tests, later call test viterbi
	
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

	std::vector<unsigned int> line_x(img.width());

	clock_t start = clock();
	viterbiLineOpenCL_cols(img_out.get(), img.height(), img.width(), &line_x[0], G_LOW, G_HIGH, command_queue, context, device_id);
	clock_t end = clock();
	double time_ms = (double)(end - start);
	for (int i = 0; i < img.width(); i++)
	{
		img(i, (int)line_x[i], 0, 0) = 255;
		img(i, (int)line_x[i], 0, 1) = 0;
		img(i, (int)line_x[i], 0, 2) = 0;
	}
	CImgDisplay gray_disp(gray_img, "image gray");
	CImgDisplay rgb_disp(img, "image rgb1");
	printf("\nviterbi parallel time, cols version: %f ms\n", time_ms);

	//reset line_x 
	for (uint32_t i = 0; i < img.width(); i++)
	{
		line_x[i] = 0;
	}

	start = clock();
	viterbiLineDetect(img_out.get(), (unsigned int)img.height(), (unsigned int)img.width(), (unsigned int*)line_x.get(), G_LOW, G_HIGH);
	end = clock();
	time_ms = (double)(end - start);
	for (int i = 0; i < img.width(); i++)
	{
		img(i, (int)line_x[i], 0, 0) = 0;
		img(i, (int)line_x[i], 0, 1) = 255;
		img(i, (int)line_x[i], 0, 2) = 0;
	}
	printf("\nViterbi serial time: %f ms\n", time_ms);
	CImgDisplay rgb1_disp(img, "Image rgb2");

	//reset line_x 
	for (uint32_t i = 0; i < img.width(); i++)
	{
		line_x[i] = 0;
	}

	start = clock();
	viterbiLineOpenCL_rows(img_out.get(), img.height(), img.width(), &line_x[0], G_LOW, G_HIGH, command_queue, context, device_id);
	end = clock();
	time_ms = (double)(end - start);
	for (int i = 0; i < img.width(); i++)
	{
		img(i, (int)line_x[i], 0, 0) = 0;
		img(i, (int)line_x[i], 0, 1) = 0;
		img(i, (int)line_x[i], 0, 2) = 255;
	}
	printf("\nViterbi parallel time , rows version: %f ms\n", time_ms);
	CImgDisplay rgb2_disp(img, "Image rgb3");
	while (!rgb1_disp.is_closed());
	//cleanup
	int err = 0;
	err = clFlush(command_queue);
	err = clFinish(command_queue);
	err = clReleaseCommandQueue(command_queue);
	err = clReleaseContext(context);
	return 0;
}