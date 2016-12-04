#include "stdafx.h"
#include "Viterbi.h"

size_t VITERBI::readKernelFile(std::string &source_str, const std::string &fileName)
{
	std::ifstream file(fileName, std::ifstream::binary);
	std::string line;
	source_str.clear();
	size_t length = 0;
	if (file.is_open())
	{
		file.seekg(0, file.end);
		length = file.tellg();
		file.seekg(0, file.beg);
		std::unique_ptr<char> buffer(new char[length]);
		// read data as a block:
		file.read(buffer.get(), length);
		source_str = std::string(buffer.get());
		file.close();
	}
	return length;
}



void VITERBI::fixGlobalSize(size_t &global_size, const size_t &local_size)
{
	if (global_size % local_size != 0)
	{
		size_t multiple = global_size / local_size;
		++multiple;
		global_size = multiple * local_size;
	}
}


int VITERBI::viterbiLineOpenCL_rows(const unsigned char *img,
	size_t img_height,
	size_t img_width,
	unsigned int *line_x,
	int g_low, int g_high,
	cl_command_queue &command_queue,
	cl_context &context,
	cl_device_id device_id)
{
	//read kernel file
	//char *source_str = (char*)malloc(MAX_SOURCE_SIZE);]
	int err = 0;
	std::string source_str;
	size_t img_size = (img_height * img_width);
	// Load the source code containing the kernel*/
	size_t source_size = readKernelFile(source_str, VITERBI_KERNEL_FILE);
	if (source_size == 0)
	{
		return 1;
	}
	// Load the source code containing the kernel*/
	//size_t source_size = readKernelFile(source_str, VITERBI_KERNEL_FILE);	
	//size_t img_size = (img_height * img_width);
	//int err = 0;
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str[0],
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

	size_t global_size = img_height;
	size_t local_size = global_size;

	size_t VL_size = img_size;
	std::vector<float> L(VL_size, 0);
	std::vector<float> V(VL_size, 0);
	//float* L = (float*)malloc(VL_size * sizeof(float));
	//float* V = (float*)malloc(VL_size * sizeof(float));

	cl_mem cmImg = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * img_size, NULL, &err);
	err = clEnqueueWriteBuffer(command_queue, cmImg, CL_FALSE, 0, sizeof(unsigned char) * img_size, img, 0, NULL, NULL);

	cl_mem cmV = clCreateBuffer(context, CL_MEM_READ_WRITE, VL_size * sizeof(float), NULL, &err);
	cl_mem cmL = clCreateBuffer(context, CL_MEM_READ_WRITE, VL_size* sizeof(float), NULL, &err);


	err = clEnqueueWriteBuffer(command_queue, cmV, CL_FALSE, 0, sizeof(float) * VL_size, &V[0], 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, cmL, CL_FALSE, 0, sizeof(float) * VL_size, &L[0], 0, NULL, NULL);

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

	int i = 0;
	float P_max = 0;
	unsigned int x_max = 0;
	//allocate buffer x_cord
	std::vector<unsigned int> x_cord(img_width, 0);
	while (i < (img_width - 1) && !err)
	{
		err = clSetKernelArg(init_V, 3, sizeof(cl_int), (void*)&i);
		// Execute OpenCL Kernel */		
		err |= clEnqueueNDRangeKernel(command_queue, init_V, 1, NULL, &global_size, NULL, 0, NULL, NULL);

		// Copy results from the memory buffer */
		err |= clEnqueueReadBuffer(command_queue, cmV, CL_TRUE, 0,
			VL_size * sizeof(float), &V[0], 0, NULL, NULL);
		for (int column = i; column < (img_width - 1); column++)
		{
			err |= clSetKernelArg(viterbi_forward, 5, sizeof(cl_int), (void*)&column);
			// Execute OpenCL Kernel */
			err |= clEnqueueNDRangeKernel(command_queue, viterbi_forward, 1, NULL, &global_size, NULL, 0, NULL, NULL);
			// Copy results from the memory buffer */
			err |= clEnqueueReadBuffer(command_queue, cmL, CL_TRUE, 0,
				VL_size * sizeof(float), &L[0], 0, NULL, NULL);
			err |= clEnqueueReadBuffer(command_queue, cmV, CL_TRUE, 0,
				VL_size * sizeof(float), &V[0], 0, NULL, NULL);
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
		for (size_t n = (img_width - 1); n > i; n--)
		{
			x_cord[n - 1] = x_cord[n] + static_cast<unsigned int>(L[(x_cord[n] * img_width) + (n - 1)]);
		}
		// save only last pixel position
		line_x[i] = x_cord[i];
		P_max = 0;
		x_max = 0;
		++i;

	}
	if (!err)
	{
		line_x[img_width - 1] = line_x[img_width - 2];
	}
	//realase resources
	err = clReleaseKernel(viterbi_forward);
	err = clReleaseProgram(program);
	err = clReleaseMemObject(cmV);
	err = clReleaseMemObject(cmL);
	err = clReleaseMemObject(cmImg);
	//free(x_cord);
	//free(L);
	//free(V);
	//free(source_str);
	return err;
}

//template <class pix_type>
int VITERBI::viterbiLineDetect(const unsigned char *img, unsigned int img_height, unsigned int img_width, unsigned int *line_x, int g_low, int g_high)
{
	if (img == 0 && img_height > 0 && img_width > 0 && line_x == 0)
	{
		return 1;
	}
	//allocate array for viterbi algorithm
	std::vector<int> L(img_height * img_width, 0);
	std::vector<uint32_t> V(img_height * img_width, 0);

	uint32_t P_max = 0;
	uint32_t x_max = 0;
	std::vector<uint32_t> x_cord(img_width, 0);
	uint32_t max_val = 0;
	size_t i = 0;
	unsigned char pixel_value = 0;
	while (i < (img_width - 1))
	{
		// init first column with zeros
		for (size_t m = 0; m < img_height; m++)
		{
			V[(m * img_width) + i] = 0;
		}
		for (size_t n = i; n < (img_width - 1); n++)
		{
			for (size_t j = 0; j < img_height; j++)
			{
				max_val = 0;
				for (int g = g_low; g <= g_high; g++)
				{
					if ((j + g) >(int)(img_height - 1))
					{
						break;
					}
					if (j + g < 0)
					{
						continue;
					}
					int curr_id = j + g;
					pixel_value = img[((curr_id)* img_width) + n];
					if ((pixel_value + V[(img_width * curr_id) + n]) > max_val)
					{
						max_val = pixel_value + V[(img_width * curr_id) + n];
						L[(j * img_width) + n] = g;
					}
				}
				V[(j * img_width) + (n + 1)] = max_val;
			}
		}
		//find biggest cost value in last column
		for (size_t j = 0; j < img_height; j++)
		{
			if (V[(j * img_width) + (img_width - 1)] > P_max)
			{
				P_max = V[(j * img_width) + (img_width - 1)];
				x_max = j;
			}
		}
		//backwards phase - retrace the path
		x_cord[(img_width - 1)] = x_max;
		for (size_t n = (img_width - 1); n > i; n--)
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
	return 0;
}

int VITERBI::viterbiLineOpenCL_cols(const unsigned char *img,
	size_t img_height,
	size_t img_width,
	int *line_x,
	int g_low, int g_high,
	cl_command_queue &command_queue,
	cl_context &context,
	cl_device_id device_id)
{
	int err = 0;
	std::string source_str;
	size_t img_size = (img_height * img_width);
	// Load the source code containing the kernel*/
	size_t source_size = readKernelFile(source_str, VITERBI_KERNEL_FILE);
	if (source_size == 0)
	{
		return 1;
	}
	char *source_str_ptr = &source_str[0];
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str_ptr,
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
	cl_kernel viterbi_forward = clCreateKernel(program, VITERBI_FUNCTION, &err);

	size_t global_size = img_width; // maybe make it later so it can be divided by Prefered opencl device multiple

	//check available memory
	long long dev_memory = 0;
	err = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(long long), &dev_memory, NULL);
	int dev_mem = static_cast<int>(dev_memory / (1024 * 1024));//MB
	int tot_mem = static_cast<int>(((img_size * global_size * sizeof(float)) +
		(2 * img_height * img_width * sizeof(float)) +
		(2 * img_width * sizeof(int)) + (img_size * sizeof(unsigned char))) / (1024 * 1024));
	printf("\nTotal memory used %d MB\n", tot_mem);

	//handle not enough GPU memory
	if (dev_mem < tot_mem)
	{
		int mem_multiple = (int)(tot_mem / dev_mem);
		global_size = img_width / (mem_multiple + 1);
	}

	//create necessery opencl buffers
	cl_mem cmImg = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * img_size, NULL, &err);
	err = clEnqueueWriteBuffer(command_queue, cmImg, CL_FALSE, 0, sizeof(unsigned char) * img_size, img, 0, NULL, NULL);

	cl_mem cmLine_x = clCreateBuffer(context, CL_MEM_READ_WRITE, img_width * sizeof(int), NULL, &err);
	cl_mem cmX_cord = clCreateBuffer(context, CL_MEM_READ_WRITE, img_width * sizeof(int), NULL, &err);
	cl_mem cmV1 = clCreateBuffer(context, CL_MEM_READ_WRITE, img_height * img_width * sizeof(float), NULL, &err);
	cl_mem cmV2 = clCreateBuffer(context, CL_MEM_READ_WRITE, img_height * img_width * sizeof(float), NULL, &err);
	cl_mem cmL = clCreateBuffer(context, CL_MEM_READ_WRITE, img_size * global_size * sizeof(float), NULL, &err);

	//set kernel arguments
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
	//to big buffer will fail with CL_MEM_OBJECT_ALLOCATION_FAILURE - have to process it with chunks
	//not all columns at the same time, call it couple of times
	size_t start_column = 0; // each iteration add global_size untile start_column >= img_width
	err = clEnqueueWriteBuffer(command_queue, cmLine_x, CL_FALSE, 0, sizeof(int) * img_width, line_x, 0, NULL, NULL);
	while (start_column < img_width && !err)
	{
		err = clSetKernelArg(viterbi_forward, 10, sizeof(cl_int), (void*)&start_column);
		err |= clEnqueueNDRangeKernel(command_queue, viterbi_forward, 1, NULL, &global_size, NULL, 0, NULL, NULL);

		// Copy results from the memory buffer */
		err |= clEnqueueReadBuffer(command_queue, cmLine_x, CL_TRUE, 0,
			img_width * sizeof(int), line_x, 0, NULL, NULL);
		start_column += global_size;
	}
	line_x[img_width - 1] = line_x[img_width - 2];
	//realase resources
	err = clReleaseKernel(viterbi_forward);
	err = clReleaseProgram(program);
	err = clReleaseMemObject(cmLine_x);
	err = clReleaseMemObject(cmL);
	err = clReleaseMemObject(cmImg);
	err = clReleaseMemObject(cmX_cord);
	err = clReleaseMemObject(cmV1);
	err = clReleaseMemObject(cmV2);
	return CL_SUCCESS;
}