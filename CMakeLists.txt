cmake_minimum_required(VERSION 3.18)
project(SudokuSolver LANGUAGES CXX CUDA)

# CUDA architektura – dla GTX 1650 ustawiamy 75
set(CMAKE_CUDA_ARCHITECTURES 75)

# Ustawienia kompilatora
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Wymuszenie polityki CMP0104 (unikamy ostrzeżeń)
cmake_policy(SET CMP0104 NEW)

# IntelliSense + lepsze wsparcie dla edytorów
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Folder z nagłówkami (np. json.hpp)
include_directories(${PROJECT_SOURCE_DIR}/include)

# Utwórz wymagane foldery przy konfiguracji
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/input)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/output/solution)

# === Kompilacja programów ===

# Generator Sudoku
add_executable(sudoku_generator src/sudoku_generator.cpp)

# Przekształcanie Sudoku do grafu (do debug/testów)
add_executable(sudoku_to_graph src/sudoku_to_graph_json.cpp)

# Solver sekwencyjny
add_executable(seq src/seq.cpp)

# Solver OpenMP
add_executable(omp src/omp.cpp)

# Solver CUDA
add_executable(cuda src/cuda.cu)
set_target_properties(cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Program do automatycznych testów
add_executable(run_tests src/run_tests.cpp)
