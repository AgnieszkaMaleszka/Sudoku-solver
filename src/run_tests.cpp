#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <algorithm>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

json loadConfig(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Nie można otworzyć pliku konfiguracyjnego!\n";
        exit(1);
    }
    json config;
    file >> config;
    return config;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " [test|solve|both]\n";
        return 1;
    }

    string mode = argv[1];
    json config = loadConfig("../../config.json");

    int N = config["tests"];
    int gridSize = config["grid_size"];
    double emptyRatio = config["empty_ratio"];
    string populationSize = to_string(config["population"].get<int>());
    string iterations = to_string(config["iterations"].get<int>());
    string mutationChance = to_string(config["mutation"].get<double>());

    ostringstream ratioStream;
    ratioStream << fixed << setprecision(2) << emptyRatio;
    string ratioStr = ratioStream.str();
    std::replace(ratioStr.begin(), ratioStr.end(), '.', '_');  // zamiana kropki na podkreślnik
    vector<string> algorithms = config["algorithms"];
    vector<string> executables;
    vector<string> prefixes;

    for (const string& alg : algorithms) {
        if (alg == "Seq") {
            executables.push_back("seq.exe");
            prefixes.push_back("sequential");
        } else if (alg == "omp") {
            executables.push_back("omp.exe");
            prefixes.push_back("openmp");
        } else if (alg == "cuda") {
            executables.push_back("cuda.exe");
            prefixes.push_back("cuda");
        } else {
            cerr << "Nieznany algorytm w config.json: " << alg << endl;
            exit(1);
        }
    }

    cout << "Running Sudoku Solver Tests (" << mode << ")\n";
    random_device rd;

    if (mode == "test" || mode == "both") {
        for (int testNum = 1; testNum <= N; testNum++) {
            string file = "../../input/sudoku_test_" + to_string(testNum) + ".txt";
            string emptyCells = to_string(int(gridSize * gridSize * emptyRatio));
            string seed = to_string(rd());

            string cmd = "sudoku_generator.exe " + file + " " + to_string(gridSize) + " " + emptyCells + " " + seed;
            cout << "CMD: " << cmd << endl;
            system(cmd.c_str());
        }
    }

    if (mode == "solve" || mode == "both") {
        for (size_t k = 0; k < executables.size(); ++k) {
            for (int testNum = 1; testNum <= N; testNum++) {
                string inFile = "../../input/sudoku_test_" + to_string(testNum) + ".txt";
                string outFile = "../../output/solution/" + prefixes[k] + "_sudoku_test_" + to_string(testNum) + ".txt";

                string mutSafe = mutationChance;
                replace(mutSafe.begin(), mutSafe.end(), '.', '_');

                string measFile = "../../output/" + prefixes[k] + "_pop" + populationSize +
                                  "_iter" + iterations + "_mut" + mutSafe + "_emptyRatio" + ratioStr +  ".txt";

                string cmd = executables[k] + " " + inFile + " " + outFile + " " + measFile +
                             " " + populationSize + " " + iterations + " " + mutationChance;

                cout << "Solving Sudoku #" << testNum << " -> " << cmd << endl;
                system(cmd.c_str());
            }
        }
    }

    cout << "All tests finished.\n";
    return 0;
}
