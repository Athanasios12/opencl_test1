#ifndef TEST_OPENCL_H
#define TEST_OPENCL_H
/*
* Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

// *********************************************************************
// oclVectorAdd Notes:  
//
// A simple OpenCL API demo application that implements 
// element by element vector addition between 2 float arrays. 
//
// Runs computations with OpenCL on the GPU device and then checks results 
// against basic host CPU/C++ computation.
//
// Uses some 'shr' and 'ocl' functions from oclUtils and shrUtils libraries for 
// compactness, but these are NOT required libs for OpenCL developement in general.
// *********************************************************************

// common SDK header for standard utilities and system libs 
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
// Name of the file with the source code for the computation kernel
// *********************************************************************
const char* cSourceFile = "test_kernels.cl";

// Host buffers for demo
// *********************************************************************
float *srcA, *srcB, *dst;        // Host buffers for OpenCL test
float* Golden;                   // Host buffer for host golden processing cross check

// OpenCL Vars
cl_context cxGPUContext;        // OpenCL context
cl_command_queue cqCommandQueue;// OpenCL command que
cl_platform_id cpPlatform;      // OpenCL platform
cl_device_id cdDevice;          // OpenCL device
cl_program cpProgram;           // OpenCL program
cl_kernel ckKernel;             // OpenCL kernel
cl_mem cmDevSrcA;               // OpenCL device source buffer A
cl_mem cmDevSrcB;               // OpenCL device source buffer B 
cl_mem cmDevDst;                // OpenCL device destination buffer 
size_t szParmDataBytes;			// Byte size of context information
size_t szKernelLength;			// Byte size of kernel code
cl_int ciErr1, ciErr2;			// Error code var
size_t MAX_SOURCE_SIZE = 0x100000;
char *source_str = (char*)malloc(MAX_SOURCE_SIZE);
const char* cExecutableName = NULL;

// demo config vars
int iNumElements = 11444777;	// Length of float arrays to process (odd # for illustration)

// Forward Declarations
// *********************************************************************
void VectorAddHost(const float* pfData1, const float* pfData2, float* pfResult, int iNumElements);
void Cleanup();

// Main function 
// *********************************************************************
void vector_add_OpenCL()
{

	size_t szGlobalWorkSize;        // 1D var for Total # of work items
	size_t szLocalWorkSize;		    // 1D var for # of work items in the work group	

	// set and log Global and Local work size dimensions

	//Get an OpenCL platform
	ciErr1 = clGetPlatformIDs(1, &cpPlatform, NULL);

	printf("clGetPlatformID...\n");
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clGetPlatformID, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		Cleanup();
	}

	//Get the devices
	ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
	printf("clGetDeviceIDs...\n");
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clGetDeviceIDs, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		Cleanup();
	}

	//Create the context
	cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ciErr1);
	printf("clCreateContext...\n");
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clCreateContext, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		Cleanup();
	}

	// Create a command-queue
	cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ciErr1);
	printf("clCreateCommandQueue...\n");
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clCreateCommandQueue, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		Cleanup();
	}

	// Read the OpenCL kernel in from source file
	printf("oclLoadProgSource (%s)...\n", cSourceFile);
	//read kernel file
	FILE *fp;
	size_t source_size;
	/* Load the source code containing the kernel*/
	fopen_s(&fp, cSourceFile, "r");	
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	// Create the program
	cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&source_str, &source_size, &ciErr1);
	printf("clCreateProgramWithSource...\n");
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clCreateProgramWithSource, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		Cleanup();
	}


	ciErr1 = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
	printf("clBuildProgram...\n");
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		Cleanup();
	}

	// Create the kernel
	ckKernel = clCreateKernel(cpProgram, "addVectors", &ciErr1);
	printf("clCreateKernel (VectorAdd)...\n");
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		Cleanup();
	}

	// --------------------------------------------------------
	// Start Core sequence... copy input data to GPU, compute, copy results back
	size_t info = 0;
	clGetKernelWorkGroupInfo(ckKernel, cdDevice, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &szLocalWorkSize, &info);
	clGetKernelWorkGroupInfo(ckKernel, cdDevice, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &szLocalWorkSize, &info);
//	szLocalWorkSize *= 10;
	szGlobalWorkSize = iNumElements;  // rounded up to the nearest multiple of the LocalWorkSize
	size_t padding = 0;
	if (szGlobalWorkSize % szLocalWorkSize != 0)
	{
		padding = szGlobalWorkSize % szLocalWorkSize;
	}

	szGlobalWorkSize = szGlobalWorkSize + padding;
	printf("Global Work Size \t\t= %u\nLocal Work Size \t\t= %u\n# of Work Groups \t\t= %u\n\n",
		szGlobalWorkSize, szLocalWorkSize, (szGlobalWorkSize % szLocalWorkSize + szGlobalWorkSize / szLocalWorkSize));

	// Allocate and initialize host arrays 
	printf("Allocate and Init Host Mem...\n");
	srcA = (float *)malloc(sizeof(float) * szGlobalWorkSize);
	srcB = (float *)malloc(sizeof(float) * szGlobalWorkSize);
	dst = (float *)malloc(sizeof(float) * szGlobalWorkSize);
	Golden = (float *)malloc(sizeof(float) * iNumElements);
	for (int i = 0; i < szGlobalWorkSize; i++)
	{
		srcA[i] = 1;
		srcB[i] = 2;
	}

	// Allocate the OpenCL buffer memory objects for source and result on the device GMEM
	cmDevSrcA = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(float) * szGlobalWorkSize, NULL, &ciErr1);
	cmDevSrcB = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(float) * szGlobalWorkSize, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	cmDevDst = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, sizeof(float) * szGlobalWorkSize, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	printf("clCreateBuffer...\n");
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		Cleanup();
	}

	// Set the Argument values
	ciErr1 = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&cmDevSrcA);
	ciErr1 |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void*)&cmDevSrcB);
	ciErr1 |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&cmDevDst);
	ciErr1 |= clSetKernelArg(ckKernel, 3, sizeof(cl_int), (void*)&iNumElements);
	printf("clSetKernelArg 0 - 3...\n\n");
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		Cleanup();
	}

	// Asynchronous write of data to GPU device
	ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, cmDevSrcA, CL_FALSE, 0, sizeof(float) * szGlobalWorkSize, srcA, 0, NULL, NULL);
	ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmDevSrcB, CL_FALSE, 0, sizeof(float) * szGlobalWorkSize, srcB, 0, NULL, NULL);
	printf("clEnqueueWriteBuffer (SrcA and SrcB)...\n");
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		Cleanup();
	}

	// Launch kernel
	clock_t start = clock();
	ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 1, NULL, &szGlobalWorkSize, NULL, 0, NULL, NULL);
	//printf("clEnqueueNDRangeKernel (VectorAdd)...\n");
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clEnqueueNDRangeKernel, Line %u error code %d !!!\n\n", __LINE__, ciErr1);
		Cleanup();
	}

	// Synchronous/blocking read of results, and check accumulated errors
	ciErr1 = clEnqueueReadBuffer(cqCommandQueue, cmDevDst, CL_TRUE, 0, sizeof(float) * szGlobalWorkSize, dst, 0, NULL, NULL);
	//printf("clEnqueueReadBuffer (Dst)...\n\n");
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clEnqueueReadBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
		Cleanup();
	}
	clock_t end = clock();
	float mseconds = (float)(end - start);
	printf("\nVector add parallel execution time : %f ms\n", mseconds);
	//--------------------------------------------------------

	// Compute and compare results for golden-host and report errors and pass/fail
	printf("Comparing against Host/C++ computation...\n\n");
	start = clock();
	VectorAddHost((const float*)srcA, (const float*)srcB, (float*)Golden, iNumElements);
	end = clock();
	mseconds = (float)(end - start);
	printf("\nVector add serial execution time : %f ms\n", mseconds);
	// Cleanup and leave
	Cleanup();
}

void Cleanup()
{
	// Cleanup allocated objects
	printf("Starting Cleanup...\n\n");
	if (source_str)free(source_str);
	if (ckKernel)clReleaseKernel(ckKernel);
	if (cpProgram)clReleaseProgram(cpProgram);
	if (cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
	if (cxGPUContext)clReleaseContext(cxGPUContext);
	if (cmDevSrcA)clReleaseMemObject(cmDevSrcA);
	if (cmDevSrcB)clReleaseMemObject(cmDevSrcB);
	if (cmDevDst)clReleaseMemObject(cmDevDst);

	// Free host memory
	free(srcA);
	free(srcB);
	free(dst);
	free(Golden);
}

// "Golden" Host processing vector addition function for comparison purposes
// *********************************************************************
void VectorAddHost(const float* pfData1, const float* pfData2, float* pfResult, int iNumElements)
{
	int i;
	for (i = 0; i < iNumElements; i++)
	{
		pfResult[i] = pfData1[i] + pfData2[i];
	}
}
#endif
