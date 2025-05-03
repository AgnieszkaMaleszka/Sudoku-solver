#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>

using namespace std;

int gridSize;
int boxSize;

// Sprawdzenie, czy liczba nie występuje w wierszu, kolumnie, ani boksie
bool unUsedInBox(const vector<vector<int>> &grid, int rowStart, int colStart, int num) {
    for (int i = 0; i < boxSize; i++)
        for (int j = 0; j < boxSize; j++)
            if (grid[rowStart + i][colStart + j] == num)
                return false;
    return true;
}

bool unUsedInRow(const vector<vector<int>> &grid, int i, int num) {
    for (int j = 0; j < gridSize; j++)
        if (grid[i][j] == num)
            return false;
    return true;
}

bool unUsedInCol(const vector<vector<int>> &grid, int j, int num) {
    for (int i = 0; i < gridSize; i++)
        if (grid[i][j] == num)
            return false;
    return true;
}

bool checkIfSafe(const vector<vector<int>> &grid, int i, int j, int num) {
    return (unUsedInRow(grid, i, num) &&
            unUsedInCol(grid, j, num) &&
            unUsedInBox(grid, i - i % boxSize, j - j % boxSize, num));
}

void fillBox(vector<vector<int>> &grid, int row, int col) {
    vector<int> nums(gridSize);
    iota(nums.begin(), nums.end(), 1);
    random_device rd;
    mt19937 g(rd());
    shuffle(nums.begin(), nums.end(), g);

    int idx = 0;
    for (int i = 0; i < boxSize; i++) {
        for (int j = 0; j < boxSize; j++) {
            while (idx < nums.size() &&
                   (!unUsedInRow(grid, row + i, nums[idx]) ||
                    !unUsedInCol(grid, col + j, nums[idx]) ||
                    !unUsedInBox(grid, row, col, nums[idx]))) {
                idx++;
            }
            if (idx >= nums.size()) {
                fillBox(grid, row, col);
                return;
            }
            grid[row + i][col + j] = nums[idx++];
        }
    }
}

void fillDiagonal(vector<vector<int>> &grid) {
    for (int i = 0; i < gridSize; i += boxSize)
        fillBox(grid, i, i);
}

bool fillRemaining(vector<vector<int>> &grid, int i, int j) {
    if (i == gridSize)
        return true;
    if (j == gridSize)
        return fillRemaining(grid, i + 1, 0);
    if (grid[i][j] != 0)
        return fillRemaining(grid, i, j + 1);

    for (int num = 1; num <= gridSize; num++) {
        if (checkIfSafe(grid, i, j, num)) {
            grid[i][j] = num;
            if (fillRemaining(grid, i, j + 1))
                return true;
            grid[i][j] = 0;
        }
    }
    return false;
}

// Solver z licznikiem rozwiązań
bool solveWithCount(vector<vector<int>> &grid, int &count, int i = 0, int j = 0) {
    if (i == gridSize) {
        count++;
        return count > 1; // wczesne zakończenie jeśli więcej niż jedno rozwiązanie
    }
    if (j == gridSize)
        return solveWithCount(grid, count, i + 1, 0);
    if (grid[i][j] != 0)
        return solveWithCount(grid, count, i, j + 1);

    for (int num = 1; num <= gridSize; num++) {
        if (checkIfSafe(grid, i, j, num)) {
            grid[i][j] = num;
            if (solveWithCount(grid, count, i, j + 1))
                return true;
            grid[i][j] = 0;
        }
    }
    return false;
}

// Usuwanie cyfr przy zachowaniu jednoznaczności rozwiązania
void removeKDigits(vector<vector<int>> &grid, int k) {
    vector<pair<int, int>> cells;
    for (int i = 0; i < gridSize; i++)
        for (int j = 0; j < gridSize; j++)
            cells.emplace_back(i, j);

    random_device rd;
    mt19937 g(rd());
    shuffle(cells.begin(), cells.end(), g);

    int removed = 0;
    for (auto [i, j] : cells) {
        if (removed >= k) break;
        int backup = grid[i][j];
        grid[i][j] = 0;

        vector<vector<int>> copy = grid;
        int count = 0;
        solveWithCount(copy, count);
        if (count != 1) {
            grid[i][j] = backup; // przywróć jeśli nieunikalne
        } else {
            removed++;
        }
    }
}

vector<vector<int>> sudokuGenerator(int k) {
    vector<vector<int>> grid(gridSize, vector<int>(gridSize, 0));
    fillDiagonal(grid);
    fillRemaining(grid, 0, 0);
    removeKDigits(grid, k);
    return grid;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Użycie: " << argv[0] << " <plik_wyjściowy> <rozmiar> <liczba pustych komórek>\n";
        return 1;
    }

    string filename = argv[1];
    gridSize = stoi(argv[2]);
    double root = sqrt(gridSize);
    if (root != floor(root)) {
        cerr << "Rozmiar planszy musi być kwadratem liczby całkowitej (np. 4, 9, 16)\n";
        return 1;
    }
    boxSize = static_cast<int>(root);

    srand(time(0));
    int k = stoi(argv[3]);
    if (k > gridSize * gridSize) {
        cerr << "Liczba pustych komórek > rozmiar planszy\n";
        return 1;
    }

    vector<vector<int>> sudoku = sudokuGenerator(k);

    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Nie można otworzyć pliku: " << filename << endl;
        return 1;
    }

    for (const auto &row : sudoku) {
        for (int cell : row)
            file << cell << " ";
        file << endl;
    }
    file.close();

    cout << "Sudoku zapisane do pliku: " << filename << " (rozmiar: " << gridSize << "x" << gridSize << ")\n";

    return 0;
}
