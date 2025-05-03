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
#include <cuda_runtime.h>
#include <curand_kernel.h>

using namespace std;

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); exit(1);}} while(0)

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

__global__ void mutateKernel(int* colors, const int* adjacency, const int* degrees, const int* fixedValues,
                             int dim, int size, double mutationRate, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size || fixedValues[idx] != 0) return;

    int conflicts = 0;
    for (int i = 0; i < degrees[idx]; ++i) {
        int neighbor = adjacency[idx * dim + i];
        if (colors[idx] == colors[neighbor]) conflicts++;
    }

    if (conflicts > 0 && curand_uniform_double(&states[idx]) < mutationRate) {
        bool forbidden[10] = {false};
        for (int i = 0; i < degrees[idx]; ++i) {
            forbidden[colors[adjacency[idx * dim + i]]] = true;
        }
        int options[9], optionCount = 0;
        for (int v = 1; v <= dim; ++v) {
            if (!forbidden[v]) options[optionCount++] = v;
        }
        if (optionCount > 0) {
            int newVal = options[(int)(curand_uniform(&states[idx]) * optionCount)];
            colors[idx] = newVal;
        } else {
            colors[idx] = 1 + (int)(curand_uniform(&states[idx]) * dim);
        }
    }
}

__global__ void setupKernel(curandState* states, unsigned long seed, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) curand_init(seed, idx, 0, &states[idx]);
}

void mutateGPU(vector<int>& colors, const Graph& graph, double mutationRate) {
    int size = graph.size;
    int dim = graph.dim;

    int* d_colors;
    int* d_adjacency;
    int* d_degrees;
    int* d_fixed;
    curandState* d_states;

    vector<int> degrees(size);
    int maxDegree = 0;
    for (int i = 0; i < size; ++i) {
        degrees[i] = graph.adjacency[i].size();
        maxDegree = max(maxDegree, degrees[i]);
    }

    vector<int> flatAdjacency(size * maxDegree, -1);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < degrees[i]; ++j) {
            flatAdjacency[i * maxDegree + j] = graph.adjacency[i][j];
        }
    }

    CUDA_CALL(cudaMalloc(&d_colors, sizeof(int) * size));
    CUDA_CALL(cudaMalloc(&d_adjacency, sizeof(int) * size * maxDegree));
    CUDA_CALL(cudaMalloc(&d_degrees, sizeof(int) * size));
    CUDA_CALL(cudaMalloc(&d_fixed, sizeof(int) * size));
    CUDA_CALL(cudaMalloc(&d_states, sizeof(curandState) * size));

    CUDA_CALL(cudaMemcpy(d_colors, colors.data(), sizeof(int) * size, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_adjacency, flatAdjacency.data(), sizeof(int) * size * maxDegree, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_degrees, degrees.data(), sizeof(int) * size, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_fixed, graph.fixedValues.data(), sizeof(int) * size, cudaMemcpyHostToDevice));

    setupKernel<<<(size + 255)/256, 256>>>(d_states, time(NULL), size);
    mutateKernel<<<(size + 255)/256, 256>>>(d_colors, d_adjacency, d_degrees, d_fixed, dim, size, mutationRate, d_states);

    CUDA_CALL(cudaMemcpy(colors.data(), d_colors, sizeof(int) * size, cudaMemcpyDeviceToHost));

    cudaFree(d_colors);
    cudaFree(d_adjacency);
    cudaFree(d_degrees);
    cudaFree(d_fixed);
    cudaFree(d_states);
}

__global__ void evaluateKernel(const int* colors, const int* adjacency, const int* degrees, int* fitness, int dim, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    int conflicts = 0;
    for (int i = 0; i < degrees[idx]; ++i) {
        int neighbor = adjacency[idx * dim + i];
        if (colors[idx] == colors[neighbor] && idx < neighbor)
            conflicts++;
    }
    atomicAdd(fitness, conflicts);
} 

class GPUMutator {
    public:
        int* d_colors;
        int* d_adjacency;
        int* d_degrees;
        int* d_fixed;
        curandState* d_states;
        int size, dim, maxDegree;
    
        GPUMutator(const Graph& graph) {
            size = graph.size;
            dim = graph.dim;
    
            vector<int> degrees(size);
            maxDegree = 0;
            for (int i = 0; i < size; ++i) {
                degrees[i] = graph.adjacency[i].size();
                maxDegree = max(maxDegree, degrees[i]);
            }
    
            vector<int> flatAdj(size * maxDegree, -1);
            for (int i = 0; i < size; ++i)
                for (int j = 0; j < degrees[i]; ++j)
                    flatAdj[i * maxDegree + j] = graph.adjacency[i][j];
    
            CUDA_CALL(cudaMalloc(&d_colors, sizeof(int) * size));
            CUDA_CALL(cudaMalloc(&d_adjacency, sizeof(int) * size * maxDegree));
            CUDA_CALL(cudaMalloc(&d_degrees, sizeof(int) * size));
            CUDA_CALL(cudaMalloc(&d_fixed, sizeof(int) * size));
            CUDA_CALL(cudaMalloc(&d_states, sizeof(curandState) * size));
    
            CUDA_CALL(cudaMemcpy(d_adjacency, flatAdj.data(), sizeof(int) * size * maxDegree, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(d_degrees, degrees.data(), sizeof(int) * size, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(d_fixed, graph.fixedValues.data(), sizeof(int) * size, cudaMemcpyHostToDevice));
    
            setupKernel<<<(size + 255)/256, 256>>>(d_states, time(NULL), size);
        }
    
        void mutate(vector<int>& host_colors, double mutationRate) {
            CUDA_CALL(cudaMemcpy(d_colors, host_colors.data(), sizeof(int) * size, cudaMemcpyHostToDevice));
            mutateKernel<<<(size + 255)/256, 256>>>(d_colors, d_adjacency, d_degrees, d_fixed, dim, size, mutationRate, d_states);
            CUDA_CALL(cudaMemcpy(host_colors.data(), d_colors, sizeof(int) * size, cudaMemcpyDeviceToHost));
        }
    
        ~GPUMutator() {
            cudaFree(d_colors);
            cudaFree(d_adjacency);
            cudaFree(d_degrees);
            cudaFree(d_fixed);
            cudaFree(d_states);
        }
    };
    
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
                    for (int v = 1; v <= graph.dim; ++v)
                        if (!used[v]) options.push_back(v);
                    if (!options.empty()) {
                        colors[idx] = options[rng() % options.size()];
                    } else {
                        colors[idx] = 1 + (rng() % graph.dim);
                    }
                }
            }
        }
    
        void evaluate(const Graph& graph) {
            fitness = 0;
            vector<bool> visited(graph.size, false);
            for (int u = 0; u < graph.size; ++u) {
                for (int v : graph.adjacency[u]) {
                    if (u < v && colors[u] == colors[v])
                        fitness++;
                }
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
    };
    
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
            vector<Individual> bufferPopulation(populationSize, Individual(graph.size));
            GPUMutator gpu(graph);
    
            #pragma omp parallel for
            for (int i = 0; i < populationSize; ++i) {
                mt19937 thread_rng(rng());
                population[i].initialize(graph, thread_rng);
                population[i].evaluate(graph);
            }
    
            int generation = 0;
            while (generation < maxGenerations) {
                sort(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
                    return a.fitness < b.fitness;
                });
    
                if (population[0].fitness == 0) break;
    
                bufferPopulation[0] = population[0];
    
                #pragma omp parallel for
                for (int i = 1; i < populationSize; ++i) {
                    mt19937 thread_rng(rng());
                    Individual parent1 = tournamentSelection(population, 3);
                    Individual parent2 = tournamentSelection(population, 3);
                    Individual child = Individual::crossover(parent1, parent2, graph, thread_rng);
                    gpu.mutate(child.colors, mutationRate);
                    child.evaluate(graph);
                    bufferPopulation[i] = move(child);
                }
    
                population.swap(bufferPopulation);
                generation++;
            }
            return population[0];
        }
    };
    
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
    ofstream measurmentOutFile(measurementFile, ios::app);

    if (solution.fitness == 0) {
        measurmentOutFile << dim <<" "<< duration.count() << endl;
        printSudoku(solution, dim, outFile);
        cout << "Solution found in " << duration.count() << " seconds." << endl;
    } else {
        cout << "No solution found." << endl;
    }

    return 0;
}