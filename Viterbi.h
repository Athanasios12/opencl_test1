#ifndef VITERBI_H
#define VITERBI_H

#include <stdio.h>
#include <CL/cl.h>
#include "CImg.h"
#include <vector>
#include <memory>
#include <fstream>
#include <string>

using namespace cimg_library;

const size_t MAX_SOURCE_SIZE = 0x100000;
const char VITERBI_KERNEL_FILE[] = "viterbi_kernel.cl";
const char VITERBI_FORWARD_FUNCTION[] = "viterbi_forward";
const char VITERBI_FUNCTION[] = "viterbi_function";
const char VITERBI_INIT_V_FUNCTION[] = "initV";

namespace VITERBI
{
	size_t readKernelFile(std::string &source_str, const std::string &fileName);
	void fixGlobalSize(size_t &global_size, const size_t &local_size);
	int viterbiLineOpenCL_rows(const unsigned char *img,
		size_t img_height,
		size_t img_width,
		unsigned int *line_x,
		int g_low, int g_high,
		cl_command_queue &command_queue,
		cl_context &context,
		cl_device_id device_id);

	//add std thread viterbi implementation on cpu
	int viterbiLineDetect(const unsigned char *img, int img_height, int img_width, std::vector<unsigned int> &line_x, int g_low, int g_high);
	int viterbiLineOpenCL_cols(const unsigned char *img,
		size_t img_height,
		size_t img_width,
		unsigned int *line_x,
		int g_low, int g_high,
		cl_command_queue &command_queue,
		cl_context &context,
		cl_device_id device_id);
}

#endif //VITERBI_H