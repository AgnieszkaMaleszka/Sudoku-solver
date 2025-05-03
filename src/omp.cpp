#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cmath>
#include <string>
#include <ctime>
#include <omp.h> // Dodane OpenMP

using namespace std;

// -------------------------
// Klasa Grafu
// -------------------------
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
                    if (k != j) adjacency[idx].push_back(i * dim + k);
                    if (k != i) adjacency[idx].push_back(k * dim + j);
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

// -------------------------
// Klasa Osobnika
// -------------------------
class Individual {
public:
    vector<int> colors;
    int fitness;
    
    Individual(int size) : colors(size), fitness(0) {}

    void initialize(const Graph& graph, mt19937& rng) {
        for (int idx = 0; idx < graph.size; ++idx) {
            if (graph.fixedValues[idx] != 0) {
                colors[idx] = graph.fixedValues[idx];
            } else {
                vector<bool> used(graph.dim + 1, false);
                for (int neighbor : graph.adjacency[idx]) {
                    int color = graph.fixedValues[neighbor];
                    if (color > 0) used[color] = true;
                }
                vector<int> options;
                for (int v = 1; v <= graph.dim; ++v) {
                    if (!used[v]) options.push_back(v);
                }
                if (!options.empty()) {
                    uniform_int_distribution<int> dist(0, options.size() - 1);
                    colors[idx] = options[dist(rng)];
                } else {
                    uniform_int_distribution<int> dist(1, graph.dim);
                    colors[idx] = dist(rng);
                }
            }
        }
    }

    void evaluate(const Graph& graph) {
        fitness = 0;
        vector<bool> visited(graph.size, false);
        for (int u = 0; u < graph.size; ++u) {
            for (int v : graph.adjacency[u]) {
                if (!visited[v] && colors[u] == colors[v])
                    fitness++;
            }
            visited[u] = true;
        }
    }

    static Individual crossover(const Individual& parent1, const Individual& parent2, const Graph& graph, mt19937& rng) {
        Individual child(graph.size);
        for (int i = 0; i < graph.size; ++i) {
            if (graph.fixedValues[i] != 0) {
                child.colors[i] = graph.fixedValues[i];
            } else {
                int conflicts1 = 0, conflicts2 = 0;
                for (int neighbor : graph.adjacency[i]) {
                    if (parent1.colors[i] == parent1.colors[neighbor]) conflicts1++;
                    if (parent2.colors[i] == parent2.colors[neighbor]) conflicts2++;
                }
                child.colors[i] = (conflicts1 <= conflicts2) ? parent1.colors[i] : parent2.colors[i];
            }
        }
        return child;
    }

    void mutate(const Graph& graph, mt19937& rng, double mutationRate) {
        uniform_real_distribution<double> dist(0.0, 1.0);
        vector<int> conflictCount(graph.size, 0);

        for (int i = 0; i < graph.size; ++i) {
            for (int neighbor : graph.adjacency[i]) {
                if (colors[i] == colors[neighbor])
                    conflictCount[i]++;
            }
        }

        for (int i = 0; i < graph.size; ++i) {
            if (graph.fixedValues[i] == 0 && conflictCount[i] > 0 && dist(rng) < mutationRate) {
                vector<bool> forbidden(graph.dim + 1, false);
                for (int neighbor : graph.adjacency[i]) {
                    forbidden[colors[neighbor]] = true;
                }
                vector<int> options;
                for (int v = 1; v <= graph.dim; ++v) {
                    if (!forbidden[v]) options.push_back(v);
                }
                if (!options.empty()) {
                    uniform_int_distribution<int> valueDist(0, options.size() - 1);
                    colors[i] = options[valueDist(rng)];
                } else {
                    uniform_int_distribution<int> valueDist(1, graph.dim);
                    colors[i] = valueDist(rng);
                }
            }
        }
    }
};

// -------------------------
// Algorytm Genetyczny
// -------------------------
class GeneticAlgorithm {
public:
    Graph graph;
    int populationSize;
    double mutationRate;
    int maxGenerations;
    mt19937 rng;

    GeneticAlgorithm(const Graph& g, int popSize, double mutRate, int maxGen)
        : graph(g), populationSize(popSize), mutationRate(mutRate), maxGenerations(maxGen) {
        rng.seed(chrono::steady_clock::now().time_since_epoch().count());
    }

    Individual tournamentSelection(const vector<Individual>& population, int tournamentSize) {
        uniform_int_distribution<int> dist(0, population.size() - 1);
        Individual best = population[dist(rng)];
        for (int i = 1; i < tournamentSize; ++i) {
            Individual challenger = population[dist(rng)];
            if (challenger.fitness < best.fitness)
                best = challenger;
        }
        return best;
    }

    Individual run() {
        vector<Individual> population(populationSize, Individual(graph.size));

        // Inicjalizacja osobników - RÓWNOLEGŁA
        #pragma omp parallel for default(none) shared(population, graph) private(i)
        for (int i = 0; i < populationSize; ++i) {
            mt19937 thread_rng(rng()); // osobny generator per wątek
            population[i].initialize(graph, thread_rng);
            population[i].evaluate(graph);
        }

        int generation = 0;
        while (generation < maxGenerations) {
            sort(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
                return a.fitness < b.fitness;
            });

            if (population[0].fitness == 0) {
                break;
            }

            vector<Individual> newPopulation(populationSize, Individual(graph.size));
            newPopulation[0] = population[0];

            // Tworzenie dzieci - RÓWNOLEGŁE
            #pragma omp parallel for
            for (int i = 1; i < populationSize; ++i) {
                mt19937 thread_rng(rng()); // osobny generator na wątek
                Individual parent1 = tournamentSelection(population, 3);
                Individual parent2 = tournamentSelection(population, 3);

                Individual child = Individual::crossover(parent1, parent2, graph, thread_rng);
                child.mutate(graph, thread_rng, mutationRate);
                child.evaluate(graph);

                newPopulation[i] = move(child);
            }

            population = move(newPopulation);
            generation++;
        }

        return population[0];
    }
};

// -------------------------
// Funkcje pomocnicze
// -------------------------
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

void printSudoku(const Individual& indiv, int dim, ofstream& outFile) {
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            outFile << indiv.colors[i * dim + j] << " ";
        }
        outFile << endl;
    }
}

// -------------------------
// Główna funkcja
// -------------------------
int main(int argc, char* argv[]) {
    if (argc != 7) {
        cout << "Usage: " << argv[0] << " <input_file> <output_file> <measurement_file> <population_size> <iterations> <mutation_rate>" << endl;
        return 1;
    }

    string inputFile = argv[1];
    string outputFile = argv[2];
    string measurementFile = argv[3];
    int populationSize = stoi(argv[4]);
    int iterations = stoi(argv[5]);
    double mutationRate = stod(argv[6]);

    int dim;
    vector<int> board = loadSudoku(inputFile, dim);

    Graph graph(board, dim);
    GeneticAlgorithm ga(graph, populationSize, mutationRate, iterations);

    auto startTime = chrono::high_resolution_clock::now();
    Individual solution = ga.run();
    auto endTime = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = endTime - startTime;

    ofstream outFile(outputFile);
    ofstream measurementOutFile(measurementFile, ios::app);

    if (solution.fitness == 0) {
        measurementOutFile << dim << " " << duration.count() << endl;
        printSudoku(solution, dim, outFile);
        cout << "Solution found in " << duration.count() << " seconds." << endl;
    } else {
        cout << "No solution found." << endl;
    }

    return 0;
}
