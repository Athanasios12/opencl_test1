#include "Tests.h"
#include <algorithm>
#include <string>

int main(int argc, char **argv)
{
	TestSettings settings;
	Algorithm algType = GPU;
#ifdef _DEBUG
	//algType = HYBRID;
	PlotInfo pInfo;
    if (!readConfig(true, settings, algType))
    {
        print("error reading file");
    }
	//test_viterbi(settings, pInfo, algType);
    print("Entering debugg");
	basicTest(settings);
#else
	if (argc < 2)
	{
		cout << "No arguments, passed\n " << 
				"Possible combinations: 0-ALL(default), 1-SERIAL, 2-THREADS, 8-GPU, 16-HYBRID\n" << 
				"Can pass combos, ex. 9 is SERIAL + GPU, 15 i SERIAL, THREADS, GPU\n";
		cout << "Since no argument passed, all algorithms will be tested, proceed?[Y/N]" << endl;
		std::string response;
		while (true)
		{
			cin >> response;
			std::transform(response.begin(), response.end(), response.begin(), ::tolower);
			if (response.compare("y") == 0)
			{
				algType = ALL;
				break;
			}
			else if (response.compare("n") == 0)
			{
				cout << "Exiting program" << endl;
				return 0;
			}
			cout << "Wrong response type try again." << endl;
		}
	}
	else if(argc == 2)
	{
		algType = static_cast<Algorithm>(std::stoi(std::string(argv[1])));
	}
	PlotInfo pInfo;
    if (!readConfig(false, settings, algType))
    {
        print("Error reading configuration file!");
    }
	test_viterbi(settings, pInfo, algType);
	generateCsv(pInfo, settings);
#endif
	return 0;
}