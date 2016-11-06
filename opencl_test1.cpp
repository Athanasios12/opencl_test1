// opencl_test1.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


#include <stdio.h>
#include <CL/cl.h>
#include <stdlib.h>
#include "CImg.h"
#include <time.h>
#include "test_opencl.h"

using namespace cimg_library;

const size_t MEM_SIZE = 10000000;
//const size_t MAX_SOURCE_SIZE = 0x100000;
const char VITERBI_KERNEL_FILE[] = "viterbi_kernel.cl";
const char VITERBI_INIT_FILE[] = "init_V_kernel.cl";
const char VITERBI_FORWARD_FUNCTION[] = "viterbi_forward";
const char VITERBI_INIT_V_FUNCTION[] = "initV";

size_t readKernelFile(char *source_str, const char *fileName)
{
	//read kernel file
	FILE *fp;
	size_t source_size;

	/* Load the source code containing the kernel*/
	fopen_s(&fp, fileName, "r");
	if (!fp) {
		return 0;
	}
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);
	return source_size;
}

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

int vectorAdd()
{
	//read kernel file	
	char *source_str = (char*)malloc(MAX_SOURCE_SIZE);
	size_t source_size = readKernelFile(source_str, "test_kernels.cl");
	//char *initVsrc_str = (char*)malloc(MAX_SOURCE_SIZE);
	//size_t initV_size = readKernelFile(initVsrc_str, VITERBI_INIT_FILE);

	//getting number of available platforms
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
	cl_device_id device_id = NULL;
	cl_uint numOfDevices = 0;

	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &numOfDevices);
	if (CL_SUCCESS == err)
	{
		printf("\nSelected device id :%d", device_id);
	}
	else
	{
		printf("\nError calling getdeviceIds. error code :%d", err);
		getchar();
		return 0;
	}

	//creating context
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
	if (CL_SUCCESS == err)
	{
		printf("\nCreated context");
	}
	else
	{
		printf("\nError creating context. error code :%d", err);
		getchar();
		return 0;
	}

	//creating command queue 
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &err);
	if (CL_SUCCESS == err)
	{
		printf("\nCreated command queue");
	}
	else
	{
		printf("\nError creating command queue. error code :%d", err);
		getchar();
		return 0;
	}

	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
		(const size_t *)&source_size, &err);
	//cl_program initV_program = clCreateProgramWithSource(context, 1, (const char **)&initVsrc_str,
	//	(const size_t *)&initV_size, &err);

	/* Build Kernel Program */
	err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	//err = clBuildProgram(initV_program, 1, &device_id, NULL, NULL, NULL);

	/* Create OpenCL Kernel */
	
	cl_kernel kernel = clCreateKernel(program, "addVectors", &err);
	cl_kernel kernel_initV = clCreateKernel(program, VITERBI_INIT_V_FUNCTION, &err);
	//allocating memory on device
	// vector sum size
	float *vector1 = (float*)malloc(sizeof(float) * MEM_SIZE);
	float *vector2 = (float*)malloc(sizeof(float) * MEM_SIZE);
	for (int i = 0; i < MEM_SIZE; i++)
	{
		vector1[i] = 1;
		vector2[i] = 2;
	}
	float *result = (float*)malloc(sizeof(float) * MEM_SIZE);

	//cl_mem mem_buff = clCreateBuffer(context, CL_MEM_READ_WRITE, MEM_SIZE * sizeof(float), NULL, &err);
	cl_mem cmVector1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MEM_SIZE, NULL, &err);
	cl_mem cmVector2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MEM_SIZE, NULL, &err);
	cl_mem cmResult = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MEM_SIZE, NULL, &err);

	size_t global_size = MEM_SIZE;
	size_t local_size = global_size;
	size_t work_size = 0, info = 0, padding = 0;
	clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &work_size, &info);
	if (global_size >= work_size)
	{
		local_size = work_size;
		if (global_size % work_size != 0)
		{
			padding = global_size % work_size;
		}
	}
	global_size = global_size + padding;
	err = clEnqueueWriteBuffer(command_queue, cmVector1, CL_FALSE, 0, sizeof(float) * MEM_SIZE, vector1, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, cmVector2, CL_FALSE, 0, sizeof(float) * MEM_SIZE, vector2, 0, NULL, NULL);

	/* Set OpenCL Kernel Parameters */
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&cmVector1);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&cmVector2);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&cmResult);
	err = clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&MEM_SIZE);
	clock_t start = clock();
	/* Execute OpenCL Kernel */
	err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("\nerror!!\n");
		return -1;
	}

	err = clEnqueueReadBuffer(command_queue, cmResult, CL_TRUE, 0,
		MEM_SIZE * sizeof(float), result, 0, NULL, NULL);
	clock_t end = clock();
	float time = (float)(end - start);	
	printf("\nVector add parallel execution time : %f s\n", time);
	start = clock();
	for (int i = 0; i < MEM_SIZE; i++)
	{
		result[i] = vector1[i] + vector2[i];
	}
	end = clock();
	time = (float)(end - start);
	printf("\nVector add serial execution time : %f s\n", time);
	//img test
	CImg<unsigned char> Img("line_3.bmp");
	//img.resize(img.height() / 2, img.width() / 2, 1);
	CImg<unsigned char> gray_img(Img.width(), Img.height(), 1, 1, 0);
	rgb2Gray(Img, gray_img);
	//viterbiLineDetect(gray_img, img, -2, 2);
	unsigned char *img_out = new unsigned char[Img.width() * Img.height()];
	memcpy(img_out, gray_img.data(0, 0, 0, 0), Img.width() * Img.height());
	size_t Img_size = Img.width() * Img.height();
	size_t img_height = Img.height();
	size_t img_width = Img.width();
	/*size_t global_size = img_height;
	size_t local_size = global_size;
	size_t work_size = 0, info = 0, padding = 0;
	clGetKernelWorkGroupInfo(kernel_initV, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &work_size, &info);
	if (global_size >= work_size)
	{
		local_size = work_size;
		if (global_size % work_size != 0)
		{
			padding = global_size % work_size;
		}
	}
	size_t VL_size = Img_size;
	global_size = global_size + padding;*/
	float *V1 = (float*)malloc(Img_size * sizeof(float));
	
	int i = 0;

	//clock_t start = clock();

	cl_mem cmV1 = clCreateBuffer(context, CL_MEM_READ_WRITE, Img_size * sizeof(float), NULL, &err);
	err = clEnqueueWriteBuffer(command_queue, cmV1, CL_FALSE, 0, sizeof(float) * Img_size, V1, 0, NULL, NULL);
	err = clSetKernelArg(kernel_initV, 0, sizeof(cl_mem), (void*)&cmV1);
	err = clSetKernelArg(kernel_initV, 1, sizeof(cl_int), (void*)&img_height);
	err = clSetKernelArg(kernel_initV, 2, sizeof(cl_int), (void*)&img_width);
	err = clSetKernelArg(kernel_initV, 3, sizeof(cl_int), (void*)&i);


	err = clEnqueueNDRangeKernel(command_queue, kernel_initV, 1, NULL, &img_height, NULL, 0, NULL, NULL);

	err = clEnqueueReadBuffer(command_queue, cmV1, CL_TRUE, 0,
		Img_size * sizeof(float), V1, 0, NULL, NULL);
	err = clFinish(command_queue);
	/*clock_t end = clock();
	float time = (float)(end - start);*/
	//display result
	
	free(V1);
	/*printf("\nresult v :\n");
	for (int i = 0; i < Img_size; i++)
	{
		printf("\n%f", V1[i]);
		V1[i] = 10;
	}*/
	//err = clFlush(command_queue);
	
	err = clReleaseKernel(kernel);
	err = clReleaseProgram(program);
	err = clReleaseMemObject(cmVector1);
	err = clReleaseMemObject(cmVector2);
	err = clReleaseMemObject(cmResult);
	err = clReleaseMemObject(cmV1);
//	err = clReleaseMemObject(cmV);
	err = clReleaseCommandQueue(command_queue);
	err = clReleaseContext(context);

	free(source_str);
	getchar();
	return 0;
}

int viterbiLineOpenCL(	const unsigned char *img, 
						size_t img_height, 
						size_t img_width,
						unsigned int *line_x, 
						int g_low, int g_high, 
						cl_command_queue &command_queue,
						cl_context &context,
						cl_device_id device_id)
{
	//read kernel file
	char *source_str = (char*)malloc(MAX_SOURCE_SIZE);
	// Load the source code containing the kernel*/
	size_t source_size = readKernelFile(source_str, VITERBI_KERNEL_FILE);	
	size_t img_size = (img_height * img_width);
	int err = 0;
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
		(const size_t *)&source_size, &err);
	if (CL_SUCCESS != err)
	{
		return 0;
	}
	// Build Kernel Program */
	err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if (err == CL_BUILD_PROGRAM_FAILURE) {
		// Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

		// Allocate memory for the log
		char *log = (char *)malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

		// Print the log
		printf("%s\n", log);
		free(log);
	}

	// Create OpenCL Kernel */
	cl_kernel viterbi_forward = clCreateKernel(program, VITERBI_FORWARD_FUNCTION, &err);
	cl_kernel init_V = clCreateKernel(program, VITERBI_INIT_V_FUNCTION, &err);
	// define another kernel for initializing V i column for each iteration
	//allocate memory for L and V matrixes
	//add padding
	size_t padding = 0;
	size_t global_size = img_height;
	size_t local_size = global_size;
	//get max local work_group size programaticly later
	size_t *work_info = (size_t*)malloc(sizeof(size_t));
	size_t ret_info = 0;
	clGetKernelWorkGroupInfo(init_V, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), work_info, &ret_info);
	printf("Work group info %d, %d", *work_info, ret_info);
	if (global_size >= *work_info)
	{
		local_size = *work_info;
		if (global_size % *work_info != 0)
		{
			padding = global_size % *work_info;
		}
	}
	size_t VL_size = img_size;
	global_size = global_size + padding;
	float* L = (float*)malloc(VL_size * sizeof(float));
	float* V = (float*)malloc(VL_size * sizeof(float));

	cl_mem cmImg = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * img_size, NULL, &err);
	err = clEnqueueWriteBuffer(command_queue, cmImg, CL_FALSE, 0, sizeof(unsigned char) * img_size, img, 0, NULL, NULL);

	cl_mem cmV = clCreateBuffer(context, CL_MEM_READ_WRITE, VL_size * sizeof(float), NULL, &err);
	cl_mem cmL = clCreateBuffer(context, CL_MEM_READ_WRITE, VL_size* sizeof(float), NULL, &err);
	int i = 0;
	unsigned long int P_max = 0;
	unsigned int x_max = 0;
	//allocate buffer x_cord
	unsigned int *x_cord = (unsigned int*)malloc(img_width * sizeof(unsigned int));

	err = clEnqueueWriteBuffer(command_queue, cmV, CL_FALSE, 0, sizeof(float) * VL_size, V, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, cmV, CL_FALSE, 0, sizeof(float) * VL_size, V, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, cmL, CL_FALSE, 0, sizeof(float) * VL_size, L, 0, NULL, NULL);
	//memset(V, 10, img_size * sizeof(unsigned long int));
	err = clSetKernelArg(init_V, 0, sizeof(cl_mem), (void*)&cmV);
	err = clSetKernelArg(init_V, 1, sizeof(cl_int), (void*)&img_height);
	err = clSetKernelArg(init_V, 2, sizeof(cl_int), (void*)&img_width);

	err = clSetKernelArg(viterbi_forward, 0, sizeof(cl_mem), (void*)&cmImg);
	err = clSetKernelArg(viterbi_forward, 1, sizeof(cl_mem), (void*)&cmL);
	err = clSetKernelArg(viterbi_forward, 2, sizeof(cl_mem), (void*)&cmV);
	err = clSetKernelArg(viterbi_forward, 3, sizeof(cl_int), (void*)&img_height);
	err = clSetKernelArg(viterbi_forward, 4, sizeof(cl_int), (void*)&img_width);
	err = clSetKernelArg(viterbi_forward, 6, sizeof(cl_int), (void*)&g_low);
	err = clSetKernelArg(viterbi_forward, 7, sizeof(cl_int), (void*)&g_high);

	while (i < (img_width - 1))
	{
		
		//err = clSetKernelArg(init_V, 0, sizeof(cl_mem), (void*)&cmV);
		//err = clSetKernelArg(init_V, 1, sizeof(cl_int), (void*)&img_height);
		//err = clSetKernelArg(init_V, 2, sizeof(cl_int), (void*)&img_width);
		err = clSetKernelArg(init_V, 3, sizeof(cl_int), (void*)&i);
		
		// Execute OpenCL Kernel */
		
		err = clEnqueueNDRangeKernel(command_queue, init_V, 1, NULL, &global_size, NULL, 0, NULL, NULL);
		/*CL_INVALID_COMMAND_QUEUE;
		CL_INVALID_KERNEL;
		CL_INVALID_CONTEXT;
		CL_INVALID_KERNEL_ARGS;
		CL_INVALID_WORK_DIMENSION;
		CL_INVALID_GLOBAL_WORK_SIZE;
		CL_INVALID_GLOBAL_OFFSET;
		CL_INVALID_WORK_GROUP_SIZE;
		CL_INVALID_WORK_ITEM_SIZE;
		CL_MISALIGNED_SUB_BUFFER_OFFSET;
		CL_INVALID_IMAGE_SIZE;
		CL_OUT_OF_RESOURCES;
		CL_OUT_OF_RESOURCES;
		CL_MEM_OBJECT_ALLOCATION_FAILURE;
		CL_INVALID_EVENT_WAIT_LIST;
		CL_OUT_OF_HOST_MEMORY;*/
		// Copy results from the memory buffer */
		err = clEnqueueReadBuffer(command_queue, cmV, CL_TRUE, 0,
			VL_size * sizeof(float), V, 0, NULL, NULL);
		//clFinish(command_queue); // this one can probably be erased later if kernels are being executed in order
		float *times = (float*)malloc(sizeof(float) * (img_width - i));
		int k = 0;
		clock_t Start = clock();
		for (int column = i; column < (img_width - 1); column++)
		{
			// Set OpenCL Kernel Parameters */
			//err = clEnqueueWriteBuffer(command_queue, cmV, CL_FALSE, 0, sizeof(float) * VL_size, V, 0, NULL, NULL);
			//err = clEnqueueWriteBuffer(command_queue, cmL, CL_FALSE, 0, sizeof(float) * VL_size, L, 0, NULL, NULL);
			//err = clSetKernelArg(viterbi_forward, 0, sizeof(cl_mem), (void*)&cmImg);
			//err = clSetKernelArg(viterbi_forward, 1, sizeof(cl_mem), (void*)&cmL);
			//err = clSetKernelArg(viterbi_forward, 2, sizeof(cl_mem), (void*)&cmV);
			//err = clSetKernelArg(viterbi_forward, 3, sizeof(cl_int), (void*)&img_height);
			//err = clSetKernelArg(viterbi_forward, 4, sizeof(cl_int), (void*)&img_width);
			clock_t start = clock();
			err = clSetKernelArg(viterbi_forward, 5, sizeof(cl_int), (void*)&column);
			//err = clSetKernelArg(viterbi_forward, 6, sizeof(cl_int), (void*)&g_low);
			//err = clSetKernelArg(viterbi_forward, 7, sizeof(cl_int), (void*)&g_high);

			// Execute OpenCL Kernel */
			err = clEnqueueNDRangeKernel(command_queue, viterbi_forward, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

			// Copy results from the memory buffer */
			err = clEnqueueReadBuffer(command_queue, cmL, CL_TRUE, 0,
				VL_size * sizeof(float), L, 0, NULL, NULL);
			err = clEnqueueReadBuffer(command_queue, cmV, CL_TRUE, 0,
				VL_size * sizeof(float), V, 0, NULL, NULL);
			clock_t end = clock();
			times[k] = (float)(end - start);
			//clFinish(command_queue);			
		}
		printf("\nViterbi execution time : \n");
		float tot_time = 0;
		for (int k = 0; k < (img_width - 1 - i); i++)
		{
			//printf("\n%.14f\n", times[k]);
			tot_time += times[k];
		}
		clock_t End = clock();
		float time = (float)(End - Start);
		printf("\nViterbi execution time : %f\n", time);
		printf("\n\n");
		free(times);
		for (int j = 0; j < img_height; j++)
		{
			if (V[(img_width * j) + (img_width - 1)] > P_max)
			{
				P_max = V[(img_width * j) + (img_width - 1)];
				x_max = j;
			}
		}
		//backwards phase - retrace the path
		x_cord[(img_width - 1)] = x_max;
		for (int n = (img_width - 1); n > i; n--)
		{
			x_cord[n - 1] = x_cord[n] + L[(x_cord[n] * img_width) + (n - 1)];
		}
		// save only last pixel position
		line_x[i] = x_cord[i];
		P_max = 0;
		x_max = 0;
		++i;

	}
	line_x[img_width - 1] = line_x[img_width - 2];
	//realase resources
	err = clReleaseKernel(viterbi_forward);
	err = clReleaseProgram(program);
	err = clReleaseMemObject(cmV);
	err = clReleaseMemObject(cmL);
	err = clReleaseMemObject(cmImg);
	free(x_cord);
	free(L);
	free(V);
	free(source_str);
	return 1;
}

template <class pix_type>
int viterbiLineDetect3(const pix_type *img, unsigned int img_height, unsigned int img_width, unsigned int *line_x, int g_low, int g_high)
{
	if (img == 0 && img_height > 0 && img_width > 0 && line_x == 0)
	{
		return 0;
	}
	//allocate array for viterbi algorithm
	float* L = (float*)malloc(img_height * img_width * sizeof(float));
	float* V = (float*)malloc(img_height * img_width * sizeof(float));

	float P_max = 0;
	unsigned int x_max = 0;
	unsigned int* x_cord = (unsigned int*)malloc(img_width * sizeof(unsigned int));
	float max_val = 0;
	int i = 0;
	pix_type pixel_value = 0;
	while (i < (img_width - 1))
	{
		// init first column with zeros
		for (int m = 0; m < img_height; m++)
		{
			V[(m * img_width) + i] = 0;
		}
		for (int n = i; n < (img_width - 1); n++)
		{
			for (int j = 0; j < img_height; j++)
			{
				max_val = 0;
				for (int g = g_low; g <= g_high; g++)
				{
					if (j + g >(img_height - 1))
					{
						break;
					}
					if (j + g < 0)
					{
						continue;
					}
					pixel_value = img[(img_width * j) + n];
					if ((pixel_value + V[((j + g) * img_width) + n]) > max_val)
					{
						max_val = pixel_value + V[((j + g) * img_width) + n];
						L[(j * img_width) + n] = g;
					}
				}
				V[(j * img_width) + (n + 1)] = max_val;
			}
		}
		//find biggest cost value in last column
		for (int j = 0; j < img_height; j++)
		{
			if (V[(j * img_width) + (img_width - 1)] > P_max)
			{
				P_max = V[(j * img_width) + (img_width - 1)];
				x_max = j;
			}
		}
		//backwards phase - retrace the path
		x_cord[(img_width - 1)] = x_max;
		for (int n = (img_width - 1); n > i; n--)
		{
			x_cord[n - 1] = x_cord[n] + L[(x_cord[n] * img_width) + (n - 1)];
		}
		// save only last pixel position
		line_x[i] = x_cord[i];
		P_max = 0;
		x_max = 0;
		++i;
	}
	line_x[img_width - 1] = line_x[img_width - 2];
	//free allocated resources
	free(V);
	free(L);
	free(x_cord);
	return 1;
}

//#pragma comment(lib, "OpenCL.lib")
int main(void)
{
	//getting number of available platforms
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
	cl_device_id device_id = NULL;
	cl_uint numOfDevices = 0;

	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &numOfDevices);
	if (CL_SUCCESS == err)
	{
		printf("\nSelected device id :%d", device_id);
	}
	else
	{
		printf("\nError calling getdeviceIds. error code :%d", err);
		getchar();
		return 0;
	}
	
	//creating context
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
	if (CL_SUCCESS == err)
	{
		printf("\nCreated context");
	}
	else
	{
		printf("\nError creating context. error code :%d", err);
		getchar();
		return 0;
	}

	//creating command queue 
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &err);
	if (CL_SUCCESS == err)
	{
		printf("\nCreated command queue");
	}
	else
	{
		printf("\nError creating command queue. error code :%d", err);
		getchar();
		return 0;
	}

	//vector_add_OpenCL();
	vectorAdd();
	CImg<unsigned char> img("line_1.bmp");
	//img.resize(img.height() / 2, img.width() / 2, 1);
	CImg<unsigned char> gray_img(img.width(), img.height(), 1, 1, 0);
	rgb2Gray(img, gray_img);
	//viterbiLineDetect(gray_img, img, -2, 2);
	unsigned char *img_out = new unsigned char[img.width() * img.height()];
	memcpy(img_out, gray_img.data(0, 0, 0, 0), img.width() * img.height());
	unsigned int *line_x = new unsigned int[img.width()];
	viterbiLineOpenCL(img_out, img.height(), img.width(), line_x, -2, 2, command_queue, context, device_id);
	//viterbiLineDetect3(img_out, img.height(), img.width(), line_x, -2, 2);
	for (int i = 0; i < img.width(); i++)
	{
		img(i, line_x[i], 0, 0) = 255;
		img(i, line_x[i], 0, 1) = 0;
		img(i, line_x[i], 0, 2) = 0;
	}
	CImgDisplay gray_disp(gray_img, "Image gray");
	CImgDisplay rgb_disp(img, "Image rgb");
	while (!rgb_disp.is_closed());
	delete[] img_out;
	delete[] line_x;
	//cleanup
	err = clFlush(command_queue);
	err = clFinish(command_queue);
	err = clReleaseCommandQueue(command_queue);
	err = clReleaseContext(context);
	return 0;
}

