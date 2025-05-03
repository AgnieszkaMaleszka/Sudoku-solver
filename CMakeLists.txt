cmake_minimum_required(VERSION 3.18)
project(SudokuSolver LANGUAGES CXX CUDA)

# === Ustawienia kompilatorów ===
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)  # GTX 1650 = architektura 75

# === Wymuszenie polityki CMP0104 (CUDA_ARCHITECTURES required) ===
cmake_policy(SET CMP0104 NEW)

# === Eksport komend dla IntelliSense / VSCode ===
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# === Folder z nagłówkami (np. nlohmann/json.hpp) ===
include_directories(${PROJECT_SOURCE_DIR}/include)

# === Foldery robocze dla input/output ===
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/input)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/output/solution)

# === Folder Release (żeby GUI znajdowało programy w build/Release/) ===
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug)

# === Programy ===

add_executable(sudoku_generator src/sudoku_generator.cpp)
add_executable(sudoku_to_graph src/sudoku_to_graph_json.cpp)
add_executable(seq src/seq.cpp)
add_executable(omp src/omp.cpp)
add_executable(cuda src/cuda.cu)
add_executable(run_tests src/run_tests.cpp)

# CUDA-specific property
set_target_properties(cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# === Dodaj include/ do wszystkich targetów ===
foreach(target sudoku_generator sudoku_to_graph seq omp cuda run_tests)
    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/include)
endforeach()
