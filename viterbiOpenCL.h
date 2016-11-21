#ifndef VITERBIOPENCL_H
#define VITERBIOPENCL_H
#include <CL/cl.h>
//constants
const size_t MAX_SOURCE_SIZE = 0x100000;
const char VITERBI_KERNEL_FILE[] = "viterbi_kernel.cl";
const char VITERBI_FORWARD_FUNCTION[] = "viterbi_forward";
const char VITERBI_FORWARD2_FUNCTION[] = "viterbi_forward2";
const char VITERBI_INIT_V_FUNCTION[] = "initV";
//globals
cl_uint numPlatforms;
cl_platform_id platform_id = NULL;
cl_device_id device_id = NULL;
cl_uint numOfDevices = 0;
cl_context context;
cl_command_queue command_queue;

int initOpenCL();

size_t readKernelFile(char *source_str, const char *fileName);

int viterbiLineOpenCL_rows(const unsigned char *img,
	size_t img_height,
	size_t img_width,
	unsigned int *line_x,
	int g_low, int g_high,
	cl_command_queue &command_queue,
	cl_context &context,
	cl_device_id device_id);

template <class pix_type>
int viterbiSerialLineDetect(const pix_type *img, unsigned int img_height, unsigned int img_width, unsigned int *line_x, int g_low, int g_high);

int viterbiLineOpenCL_cols(const unsigned char *img,
	size_t img_height,
	size_t img_width,
	int *line_x,
	int g_low, int g_high,
	cl_command_queue &command_queue,
	cl_context &context,
	cl_device_id device_id);

void openCL_cleanup();
#endif //VITERBIOPENCL_H
