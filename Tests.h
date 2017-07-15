#include <stdio.h>
#include <CL/cl.h>
#include "CImg.h"
#include <time.h>
#include <memory>
#include <string>
#include <vector>
#include "Viterbi.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <stdlib.h>
#include <sstream>
#include "rapidxml\rapidxml.hpp"

#ifdef _DEBUG
#define print(x) cout << x << endl;
#else
#define print(x)
#endif // _DEBUG

using namespace cimg_library;
using namespace std;
using namespace rapidxml;

//image settings
const char CONFIG_FILE[] = "viterbi_config.xml";
const char DEBUG_SETTINGS_NODE[] = "DebugSettings";
const char RELEASE_SETTINGS_NODE[] = "ReleaseSettings";
const char DEBUG_IMG_NODE[] = "img_in";
const char DEBUG_RESULT_NODE[] = "result";
const char TESTFILES_NODE[] = "TestFiles";
const char GLOW_NODE[] = "G_LOW";
const char GHIGH_NODE[] = "G_HIGH";
const char GINCR_NODE[] = "G_INCR";
const char LINEWIDTH_NODE[] = "LINE_WIDTH";
const char IMG_NODE[] = "img";
const char IMG_NAME[] = "name";
const char IMG_TESTLINE[] = "test_line";
const char GRANGE_NODE[] = "GRange";
const char CSV_NODE[] = "CSV";
const char RESULT_COLUMNS[] = "result_columns";
const char COLUMNS_NODE[] = "column";

enum Algorithm
{
	ALL = 0,
	SERIAL = 1,
	THREADS = 2,
	GPU = 8,
	HYBRID = 16,
};

typedef struct
{
	uint32_t m_img_num;
	std::string m_img_name;
	double m_img_size;
	int m_g_low;
	int m_g_high;
	std::vector<double> m_exec_time;
	std::vector<std::vector<unsigned int> > m_lines_pos;
	uint32_t m_detectionError;
}PlotData;

typedef struct
{
	std::vector<PlotData> m_pData;
	uint32_t m_plot_id;
}PlotInfo;

struct Color
{
	uint8_t R = 255;
	uint8_t G = 0;
	uint8_t B = 0;
	Color() :R(255), G(0), B(0){}
};

typedef struct
{
	int g_high;
	int g_low;
	int g_incr;
	int line_width;
	std::vector<std::string> img_names;
	std::vector<std::string> test_lines;
	std::string csvFile;
	std::vector<std::string> columns;

}TestSettings;

bool readConfig(bool debugMode, TestSettings &settings, Algorithm algType)
{
	xml_document<> doc;
	std::ifstream file(CONFIG_FILE);
	std::stringstream buffer;
	bool success = false;
	if (file.is_open())
	{
		buffer << file.rdbuf();
		file.close();
		std::string content(buffer.str());
		doc.parse<0>(&content[0]);

		//xml parsing
		xml_node<> *pRoot = doc.first_node();
		for (xml_node<> *mainNode = doc.first_node(); mainNode; mainNode = mainNode->next_sibling())
		{
			if (debugMode && (std::string(mainNode->name()).compare(DEBUG_SETTINGS_NODE) == 0))
			{
				for (xml_node<> *pNode = mainNode->first_node(); pNode; pNode = pNode->next_sibling())
				{
					std::string nodeName = pNode->name();
					if (nodeName.compare(TESTFILES_NODE) == 0)
					{
						for (xml_node<> *imgPtr = pNode->first_node(); imgPtr; imgPtr = imgPtr->next_sibling())
						{
							settings.img_names.push_back(imgPtr->value());
						}
					}
					else if (nodeName.compare(GLOW_NODE) == 0)
					{
						settings.g_low = std::stoi(pNode->value());
					}
					else if (nodeName.compare(GHIGH_NODE) == 0)
					{
						settings.g_high = std::stoi(pNode->value());
					}
					else if (nodeName.compare(LINEWIDTH_NODE) == 0)
					{
						settings.line_width = std::stoi(pNode->value());
					}
				}
			}
			else if (!debugMode && (std::string(mainNode->name()).compare(RELEASE_SETTINGS_NODE) == 0))
			{
				for (xml_node<> *pNode = mainNode->first_node(); pNode; pNode = pNode->next_sibling())
				{
					std::string nodeName = pNode->name();
					if (nodeName.compare(TESTFILES_NODE) == 0)
					{
						for (xml_node<> *imgPtr = pNode->first_node(); imgPtr; imgPtr = imgPtr->next_sibling())
						{
							std::string name = imgPtr->name();
							if (name.compare(IMG_NODE) == 0)
							{
								for (xml_node<> *imgInfo = imgPtr->first_node(); imgInfo; imgInfo = imgInfo->next_sibling())
								{
									std::string info = imgInfo->name();
									if (info.compare(IMG_NAME) == 0)
									{
										settings.img_names.push_back(imgInfo->value());
									}
									if (info.compare(IMG_TESTLINE) == 0)
									{
										settings.test_lines.push_back(imgInfo->value());
									}
								}
							}
						}
					}
					if (nodeName.compare(GRANGE_NODE) == 0)
					{
						for (xml_node<> *gPtr = pNode->first_node(); gPtr; gPtr = gPtr->next_sibling())
						{
							std::string gName = gPtr->name();
							if (gName.compare(GLOW_NODE) == 0)
							{
								settings.g_low = std::stoi(gPtr->value());
							}
							if (gName.compare(GHIGH_NODE) == 0)
							{
								settings.g_high = std::stoi(gPtr->value());
							}
							if (gName.compare(GINCR_NODE) == 0)
							{
								settings.g_incr = std::stoi(gPtr->value());
							}
						}
					}
					if (nodeName.compare(CSV_NODE) == 0)
					{
						settings.csvFile = pNode->value();
					}
					if (nodeName.compare(LINEWIDTH_NODE) == 0)
					{
						settings.line_width = std::stoi(pNode->value());
					}
					if (nodeName.compare(RESULT_COLUMNS) == 0)
					{
						for (xml_node<> *columnPtr = pNode->first_node(); columnPtr; columnPtr = columnPtr->next_sibling())
						{
							std::string columnName = columnPtr->name();
							if (columnName.compare(COLUMNS_NODE) == 0)
							{
								bool hasId = false;
								if (algType != ALL)
								{
									for (xml_attribute<> *id = columnPtr->first_attribute(); id; id = id->next_attribute())
									{
										std::string id_name = id->name();
										int id_val = std::stoi(id->value());
										if (id_name.compare("id") == 0)
										{
											if ((static_cast<int>(algType)& id_val) != 0 || id_val == 0)
											{
												settings.columns.push_back(columnPtr->value());
											}
										}
									}
								}
								else
								{
									settings.columns.push_back(columnPtr->value());
								}
							}

						}
					}
				}
			}
		}
		return true;
	}
	return false;
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

void displayTrackedLine(CImg<unsigned char> &img, const std::vector<unsigned int> &line_x, int line_width, Color color)
{
	for (int i = 0; i < img.width(); i++)
	{
		for (int j = -line_width; j <= line_width; j++)
		{
			int idx = (int)line_x[i] + j;
			if (idx < 0)
			{
				continue;
			}
			else if (idx > img.height() - 1)
			{
				break;
			}
			img(i, idx, 0, 0) = color.R;
			img(i, idx, 0, 1) = color.G;
			img(i, idx, 0, 2) = color.B;
		}
	}
}

bool readTestLine(const std::string &fileName, std::vector<uint32_t> &test_line)
{
	std::ifstream test_file(fileName);
	std::string line;
	if (test_file.is_open())
	{
		test_line.clear();
		while (std::getline(test_file, line))
		{
			test_line.push_back(static_cast<uint32_t>(std::stoi(line)));
		}
		test_file.close();
	}
	else
	{
		return false;
	}
	return true;
}

uint32_t checkDetectionError(const std::vector<unsigned int> &line, const std::vector<uint32_t> &test_line)
{
	uint32_t error = 0;
	if (line.size() == test_line.size())
	{
		for (uint32_t i = 0; i < test_line.size(); i++)
		{
			error += uint32_t(abs(int(test_line[i]) - int(line[i])));
		}
	}
	return error;
}

bool test_viterbi(const TestSettings &settings, PlotInfo &plotInfo, Algorithm algType)
{
	if (settings.g_high <= settings.g_low || settings.g_high < 0 || settings.g_low > 0)
	{
		return false;
	}
	std::vector<std::vector<uint32_t> > test_lines;
	std::vector<uint32_t> test_line;
	bool success = true;
	for (auto && name : settings.test_lines)
	{
		success = readTestLine(name, test_line);
		test_lines.push_back(test_line);
	}
	if (!success)
	{
		return false;
	}
	cl_device_id device_id = NULL;
	cl_command_queue command_queue = NULL;
	cl_context context = NULL;
	//initalize OpenCL 
	if (CL_SUCCESS != initializeCL(command_queue, context, device_id))
	{
		printf("\nFailed OpenCl initialization!\n");
		return false;
	}
	Viterbi viterbi(command_queue, context, device_id);
	uint32_t img_num = 0;
	for (auto && img_file : settings.img_names)
	{
		CImg<unsigned char> img(img_file.c_str());
		CImg<unsigned char> img_test(img_file.c_str());
		CImg<unsigned char> gray_img(img.width(), img.height(), 1, 1, 0);
		//convert to monochrome image
		rgb2Gray(img, gray_img);
		std::unique_ptr<unsigned char> img_out(new unsigned char[img.width() * img.height()]);
		memcpy(img_out.get(), gray_img.data(0, 0, 0, 0), img.width() * img.height());

		PlotData data;
		data.m_img_name = img_file.substr(0, img_file.find("."));
		data.m_img_num = img_num;
		data.m_img_size = double(img.width() * img.height()) / double(1024 * 1024);

		viterbi.setImg(gray_img, img.height(), img.width());
		uint8_t n = 0;
		int g_low = settings.g_low;
		int g_high = settings.g_high;
		std::vector<unsigned int> line_x(img.width(), 0);
		//prepare hybrid for optimal work - test it with biggest neighbour range - prefer gpu
		viterbi.launchHybridViterbi(line_x, g_low, g_high);
		while (g_high + abs(g_low) > 0 && g_low <= 0 && g_high >= 0) // maybe redundant but whatever...
		{
			std::vector<std::vector<unsigned int> > line_results;
			std::vector<double> times;
			line_x = std::vector<unsigned int>(img.width(), 0);
			std::vector<uint32_t> detectionError;
			clock_t start, end;
			double time_ms;
			//viterbi parallel cols gpu version 
			if (algType & GPU || algType == ALL)
			{
				start = clock();
				viterbi.viterbiLineOpenCL_cols(&line_x[0], g_low, g_high);
				end = clock();
				time_ms = (double)(end - start);
				detectionError.push_back(checkDetectionError(line_x, test_lines[img_num]));
				line_results.push_back(line_x);
				times.push_back(time_ms);
				print("\nGPU time :" + std::to_string(time_ms));
			}
			//viterbi serial cpu version
			if (algType & SERIAL || algType == ALL)
			{
				start = clock();
				viterbi.viterbiLineDetect(line_x, g_low, g_high);
				end = clock();
				time_ms = (double)(end - start);
				//check line detection error based on desired cordinates from line_test.py script, save it in another column
				detectionError.push_back(checkDetectionError(line_x, test_lines[img_num]));
				line_results.push_back(line_x);
				times.push_back(time_ms);
				print("\nSerial time :" + std::to_string(time_ms));
			}
			//cpu multithread version
			if (algType & THREADS || algType == ALL)
			{
				start = clock();
				viterbi.launchViterbiMultiThread(line_x, g_low, g_high);
				end = clock();
				time_ms = (double)(end - start);
				detectionError.push_back(checkDetectionError(line_x, test_lines[img_num]));
				line_results.push_back(line_x);
				times.push_back(time_ms);
				print("\nThreads time :" + std::to_string(time_ms));
			}
			//hybrid
			if (algType & HYBRID || algType == ALL)
			{
				start = clock();
				viterbi.launchHybridViterbi(line_x, g_low, g_high);
				end = clock();
				time_ms = (double)(end - start);
				line_results.push_back(line_x);
				times.push_back(time_ms);
				print("\nHybrid time :" + std::to_string(time_ms));
			}
			//save data for ploting and tabel representation
			data.m_g_high = g_high;
			data.m_g_low = g_low;
			data.m_exec_time = times;
			data.m_lines_pos = line_results;
			uint32_t avg_err = std::accumulate(detectionError.begin(), detectionError.end(), 0) / detectionError.size();
			for (auto && error : detectionError)
			{
				print("\nError : " + std::to_string(error));
			}
			data.m_detectionError = avg_err;
			//save plot data in plot info data vector
			plotInfo.m_pData.push_back(data);
			//next g_l/h combo
			displayTrackedLine(img_test, line_x, settings.line_width, Color());
			//save img reasult
			std::string g_img_name = data.m_img_name + "_ghigh_" + std::to_string(g_high) + "_glow_" + std::to_string(g_low) + "_result.bmp";
			img_test.save_bmp(g_img_name.c_str());
			g_low += settings.g_incr;
			g_high -= settings.g_incr;
			++n;
			img_test = img;
		}
		++img_num;
	}
	//cleanup
	int err = 0;
	err |= clFlush(command_queue);
	err |= clFinish(command_queue);
	err |= clReleaseCommandQueue(command_queue);
	err |= clReleaseContext(context);
	return err == CL_SUCCESS;
}

void generateCsv(const PlotInfo &pInfo, const TestSettings &settings)
{
	std::ofstream data_file;
	data_file.open(settings.csvFile);
	if (data_file.is_open())
	{
		uint32_t i = 0;
		for (auto && column : settings.columns)
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
			data_file << data.m_img_num << "," << data.m_img_size << "," << data.m_g_low << "," << data.m_g_high << "," << data.m_detectionError;
			for (auto && time : data.m_exec_time)
			{
				data_file << "," << time;
			}
			data_file << "\n";
		}
		data_file.close();
	}
}

int basicTest(const TestSettings &settings)
{
	cl_device_id device_id = NULL;
	cl_command_queue command_queue = NULL;
	cl_context context = NULL;
	//initalize OpenCL 
	if (CL_SUCCESS != initializeCL(command_queue, context, device_id))
	{
		printf("\nFailed OpenCl initialization!\n");
		return false;
	}
	CImg<unsigned char> img(settings.img_names[0].c_str());
	CImg<unsigned char> gray_img(img.width(), img.height(), 1, 1, 0);
	//convert to monochrome image
	rgb2Gray(img, gray_img);

	std::unique_ptr<unsigned char> img_out(new unsigned char[img.width() * img.height()]);
	memcpy(img_out.get(), gray_img.data(0, 0, 0, 0), img.width() * img.height());

	std::vector<unsigned int> line_x(img.width());
	Viterbi viterbi(command_queue, context, device_id);
	viterbi.setImg(gray_img, img.height(), img.width());
	clock_t start = clock();
	//viterbi.viterbiLineOpenCL_cols(&line_x[0], settings.g_low, settings.g_high);
	clock_t end = clock();
	double time_ms = (double)(end - start);
	//displayTrackedLine(img, line_x, settings.line_width, Color());
	//CImgDisplay gray_disp(gray_img, "image gray");
	//CImgDisplay rgb_disp(img, "image rgb1");
	//printf("\nviterbi parallel time, cols version: %f ms\n", time_ms);
	line_x = std::vector<unsigned int>(img.width(), 0);
	start = clock();
	viterbi.launchHybridViterbiOpenMP(line_x, settings.g_low, settings.g_high);
	end = clock();
	time_ms = (double)(end - start);
	displayTrackedLine(img, line_x, settings.line_width, Color());
	printf("\nViterbi serial time: %f ms\n", time_ms);
	CImgDisplay rgb1_disp(img, "Image rgb2");

	start = clock();
	//viterbi.launchViterbiMultiThread(line_x, settings.g_low, settings.g_high);
	end = clock();
	time_ms = (double)(end - start);
	//displayTrackedLine(img, line_x, settings.line_width, Color());
	//printf("\nViterbi parallel time , threads CPU version: %f ms\n", time_ms);
	//CImgDisplay rgb2_disp(img, "Image rgb3");
	img.save_bmp("threds1.bmp");
	while (!rgb1_disp.is_closed());
	//cleanup
	int err = 0;
	err = clFlush(command_queue);
	err |= clFinish(command_queue);
	err |= clReleaseCommandQueue(command_queue);
	err |= clReleaseContext(context);

	return err;
}