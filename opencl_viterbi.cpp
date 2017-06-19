#include "Tests.h"

int main(void)
{
	TestSettings settings;
#ifdef _DEBUG
	readConfig(true, settings);
	basicTest(settings);
#else
	PlotInfo pInfo;
	readConfig(false, settings);
	test_viterbi(settings, pInfo);
	generateCsv(pInfo, settings);
#endif
	return 0;
}