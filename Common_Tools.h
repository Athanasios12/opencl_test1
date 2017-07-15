#ifndef COMMON_TOOLS_H
#define COMMON_TOOLS_H

#ifdef _DEBUG
#define print(x) cout << x << endl;
#else
#define print(x)
#endif // _DEBUG

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
	Color() :R(255), G(0), B(0) {}
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
#endif