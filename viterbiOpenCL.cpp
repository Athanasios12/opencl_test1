#include "viterbiOpenCL.h"

int initOpenCL()
{

	//getting number of available platforms
	cl_int err;


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
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
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
	command_queue = clCreateCommandQueue(context, device_id, 0, &err);
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
}

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

int viterbiLineOpenCL_rows(const unsigned char *img,
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
	size_t work_info = 0;
	size_t ret_info = 0;
	clGetKernelWorkGroupInfo(init_V, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &work_info, &ret_info);
	printf("Work group info %d, %d", work_info, ret_info);
	if (global_size >= work_info)
	{
		local_size = work_info;
		if (global_size % work_info != 0)
		{
			padding = global_size % work_info;
		}
	}
	size_t VL_size = img_size;
	global_size = global_size + padding;
	float* L = (float*)malloc(VL_size * sizeof(float));
	float* V = (float*)malloc(VL_size * sizeof(float));

	cl_mem cmImg = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * img_size, NULL, &err);
	err = clEnqueueWriteBuffer(command_queue, cmImg, CL_FALSE, 0, sizeof(unsigned char) * img_size, img, 0, NULL, NULL);

	cl_mem cmV = clCreateBuffer(context, CL_MEM_READ_WRITE, VL_size * sizeof(float), NULL, &err);
	cl_mem cmL = clCreateBuffer(context, CL_MEM_READ_WRITE, VL_size * sizeof(float), NULL, &err);
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
		// Copy results from the memory buffer */
		err = clEnqueueReadBuffer(command_queue, cmV, CL_TRUE, 0,
			VL_size * sizeof(float), V, 0, NULL, NULL);
		int k = 0;
		clock_t Start = clock();
		for (int column = i; column < (img_width - 1); column++)
		{
			err = clSetKernelArg(viterbi_forward, 5, sizeof(cl_int), (void*)&column);
			// Execute OpenCL Kernel */
			err = clEnqueueNDRangeKernel(command_queue, viterbi_forward, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
			// Copy results from the memory buffer */
			err = clEnqueueReadBuffer(command_queue, cmL, CL_TRUE, 0,
				VL_size * sizeof(float), L, 0, NULL, NULL);
			err = clEnqueueReadBuffer(command_queue, cmV, CL_TRUE, 0,
				VL_size * sizeof(float), V, 0, NULL, NULL);
		}

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
int viterbiSerialLineDetect(const pix_type *img, unsigned int img_height, unsigned int img_width, unsigned int *line_x, int g_low, int g_high)
{
	if (img == 0 && img_height > 0 && img_width > 0 && line_x == 0)
	{
		return 0;
	}
	//allocate array for viterbi algorithm
	float* L = (float*)malloc(img_height * img_width * img_height * sizeof(float));
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
					if ((pixel_value + V[(j * img_width) + n]) > max_val)
					{
						max_val = pixel_value + V[(j * img_width) + n];
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

int viterbiLineOpenCL_cols(const unsigned char *img,
	size_t img_height,
	size_t img_width,
	int *line_x,
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
		return err;
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
	cl_kernel viterbi_forward = clCreateKernel(program, VITERBI_FORWARD2_FUNCTION, &err);

	size_t global_size = img_width; // maybe make it later so it can be diveided by Prefered opencl device multiple

	cl_mem cmImg = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * img_size, NULL, &err);
	err = clEnqueueWriteBuffer(command_queue, cmImg, CL_FALSE, 0, sizeof(unsigned char) * img_size, img, 0, NULL, NULL);

	//float *line_x = (float*)malloc(sizeof(float) * img_width);
	//float *L = (float*)malloc(sizeof(float) * global_size * img_size);
	cl_mem cmLine_x = clCreateBuffer(context, CL_MEM_READ_WRITE, img_width * sizeof(int), NULL, &err);
	cl_mem cmX_cord = clCreateBuffer(context, CL_MEM_READ_WRITE, img_width * sizeof(int), NULL, &err);
	cl_mem cmV1 = clCreateBuffer(context, CL_MEM_READ_WRITE, img_height * img_width * sizeof(float), NULL, &err);
	cl_mem cmV2 = clCreateBuffer(context, CL_MEM_READ_WRITE, img_height * img_width * sizeof(float), NULL, &err);
	cl_mem cmL = clCreateBuffer(context, CL_MEM_READ_WRITE, img_size * global_size * sizeof(float), NULL, &err);
	//err = clEnqueueWriteBuffer(command_queue, cmLine_x, CL_FALSE, 0, sizeof(float) * img_width, line_x, NULL, NULL);
	//err = clEnqueueWriteBuffer(command_queue, cmL, CL_FALSE, 0, sizeof(float) * img_size * global_size, L, 0, NULL, NULL);

	err = clSetKernelArg(viterbi_forward, 0, sizeof(cl_mem), (void*)&cmImg);
	err |= clSetKernelArg(viterbi_forward, 1, sizeof(cl_mem), (void*)&cmL);
	err |= clSetKernelArg(viterbi_forward, 2, sizeof(cl_mem), (void*)&cmLine_x);
	err |= clSetKernelArg(viterbi_forward, 3, sizeof(cl_mem), (void*)&cmV1);
	err |= clSetKernelArg(viterbi_forward, 4, sizeof(cl_mem), (void*)&cmV2);
	err |= clSetKernelArg(viterbi_forward, 5, sizeof(cl_mem), (void*)&cmX_cord);
	err |= clSetKernelArg(viterbi_forward, 6, sizeof(cl_int), (void*)&img_height);
	err |= clSetKernelArg(viterbi_forward, 7, sizeof(cl_int), (void*)&img_width);
	err |= clSetKernelArg(viterbi_forward, 8, sizeof(cl_int), (void*)&g_high);
	err |= clSetKernelArg(viterbi_forward, 9, sizeof(cl_int), (void*)&g_low);

	if (CL_SUCCESS != err)
	{
		return err; //
	}

	err = clEnqueueNDRangeKernel(command_queue, viterbi_forward, 1, NULL, &global_size, NULL, 0, NULL, NULL);

	// Copy results from the memory buffer */
	err = clEnqueueReadBuffer(command_queue, cmLine_x, CL_TRUE, 0,
		img_width * sizeof(int), line_x, 0, NULL, NULL);

	line_x[img_width - 1] = line_x[img_width - 2];
	//realase resources
	err = clReleaseKernel(viterbi_forward);
	err = clReleaseProgram(program);
	err = clReleaseMemObject(cmLine_x);
	err = clReleaseMemObject(cmV1);
	err = clReleaseMemObject(cmV2);
	err = clReleaseMemObject(cmX_cord);
	err = clReleaseMemObject(cmImg);
	free(source_str);
	return CL_SUCCESS;
}

void openCL_cleanup()
{
	err = clFlush(command_queue);
	err = clFinish(command_queue);
	err = clReleaseCommandQueue(command_queue);
	err = clReleaseContext(context);
}