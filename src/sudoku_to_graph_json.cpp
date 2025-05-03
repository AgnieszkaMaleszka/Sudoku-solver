#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string>
#include <algorithm>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

// Klasa grafu oparta na Sudoku
class Graph {
public:
    int size;
    int dim;
    vector<vector<int>> adjacency;
    vector<int> fixedValues;

    Graph(const vector<int>& board, int dim) : dim(dim) {
        size = dim * dim;
        fixedValues = board;
        adjacency.resize(size);

        int blockSize = sqrt(dim);

        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                int idx = i * dim + j;
                for (int k = 0; k < dim; ++k) {
                    if (k != j) adjacency[idx].push_back(i * dim + k); // Wiersz
                    if (k != i) adjacency[idx].push_back(k * dim + j); // Kolumna
                }
                int blockRow = (i / blockSize) * blockSize;
                int blockCol = (j / blockSize) * blockSize;
                for (int r = 0; r < blockSize; ++r) {
                    for (int c = 0; c < blockSize; ++c) {
                        int ni = blockRow + r;
                        int nj = blockCol + c;
                        int nidx = ni * dim + nj;
                        if (nidx != idx)
                            adjacency[idx].push_back(nidx);
                    }
                }
                sort(adjacency[idx].begin(), adjacency[idx].end());
                adjacency[idx].erase(unique(adjacency[idx].begin(), adjacency[idx].end()), adjacency[idx].end());
            }
        }
    }
};

// Wczytywanie Sudoku z pliku tekstowego
vector<int> loadSudoku(const string& filename, int& dim) {
    ifstream file(filename);
    string line;
    vector<int> board;

    while (getline(file, line)) {
        istringstream iss(line);
        int num;
        while (iss >> num) {
            board.push_back(num);
        }
    }

    dim = sqrt(board.size());
    return board;
}

// Zapis grafu do JSON
void saveGraphToJson(const Graph& graph, const string& filename) {
    json j;
    j["dim"] = graph.dim;
    j["size"] = graph.size;
    j["fixedValues"] = graph.fixedValues;

    j["adjacency"] = json::array();
    for (const auto& neighbors : graph.adjacency) {
        j["adjacency"].push_back(neighbors);
    }

    ofstream file(filename);
    file << setw(4) << j;
    file.close();
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <input_sudoku_file> <output_graph_json>" << endl;
        return 1;
    }

    string sudokuFile = argv[1];
    string jsonOutputFile = argv[2];

    int dim;
    vector<int> board = loadSudoku(sudokuFile, dim);

    Graph graph(board, dim);
    saveGraphToJson(graph, jsonOutputFile);

    cout << "Graph saved to " << jsonOutputFile << endl;
    return 0;
}
