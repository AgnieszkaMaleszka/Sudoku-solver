# 🧙‍♂️ Sudoku Solver – Algorytm Kolorowania Grafu (CPU + OpenMP + CUDA)

Projekt rozwiązujący Sudoku dowolnego rozmiaru `n × n` przy pomocy **algorytmu kolorowania grafu**, zaimplementowany w trzech wersjach:

* ✅ **Sekwencyjnej (CPU)**
* 🧥 **Wielowątkowej (OpenMP)**
* ⚡ **GPU (CUDA)**

Zastosowano podejście heurystyczne z **algorytmem genetycznym**, w którym kolejne osobniki próbują pokolorować graf odpowiadający planszy Sudoku, zgodnie z jej regułami.

---

## 🔧 Funkcje projektu

* ✅ **Rozwiązywanie Sudoku** przez kolorowanie grafu
* 🌟 Trzy wersje solvera: `seq`, `omp`, `cuda`
* ⚙️ Konfigurowalne parametry: populacja, iteracje, mutacja *(można też ustawić w pliku konfiguracyjnym)*
* 🎲 **Generator plansz** z gwarantowaną jednoznacznością
* 📄 **Eksport planszy do grafu** w formacie JSON
* 🧪 **Automatyczne testy i pomiary czasu**
* 🖼️ **GUI (Tkinter)** z możliwością wyboru algorytmu

---

## 💻 Wymagania

### Kompilacja C++

* CMake ≥ 3.18
* Kompilator z C++17 (MSVC, GCC lub Clang)
* OpenMP (`omp.exe`)
* CUDA Toolkit ≥ 11.0 (`cuda.exe`)
* `nlohmann/json.hpp` w katalogu `include/`

### GUI (Python)

* Python 3.8+
* Tkinter (zwykle wbudowany)
* Opcjonalnie: `matplotlib` dla wizualizacji

---

## 🛠️ Kompilacja (Visual Studio + CUDA)

### 1. Wejdź do folderu projektu

```bash
cd SudokuSolver
```

### 2. Utwórz katalog `build` i przejdź do niego

```bash
mkdir build
cd build
```

### 3. Skonfiguruj projekt z użyciem Visual Studio (wymagane dla CUDA)

```bash
cmake .. -G "Visual Studio 17 2022" -A x64
```

> ✅ Jeśli nie masz Visual Studio 2022, zmień nazwę generatora zgodnie z Twoją wersją

### 4. Zbuduj projekt (tryb Release)

```bash
cmake --build . --config Release
```

W katalogu `build/Release/` znajdziesz:

```
seq.exe
omp.exe
cuda.exe
sudoku_generator.exe
sudoku_to_graph.exe
run_tests.exe
```

---

## 💾 Użycie solvera (przykład)

```bash
./seq.exe input.txt output.txt measurements.txt 500 1000 0.05
```

Parametry:

* `input.txt` – plik wejściowy z planszą (`0` = puste pole)
* `output.txt` – plik wyjściowy z rozwiązaniem
* `measurements.txt` – pomiar czasu
* `500` – rozmiar populacji
* `1000` – liczba iteracji
* `0.05` – współczynnik mutacji

> ℹ️ Parametry można też zdefiniować w pliku konfiguracyjnym, jeśli solver obsługuje taki tryb (np. JSON lub INI)

---

## 🎨 Uruchamianie GUI

```bash
python gui/sudoku_gui.py
```

GUI pozwala:

* Generować nowe plansze (`sudoku_generator`)
* Rozwiązywać je dowolnym algorytmem (`seq`, `omp`, `cuda`)
* Śwledzić czas działania i poprawność
* Eksperymentować z parametrami GA

---

## ✅ Przykładowe komendy

```bash
./sudoku_generator.exe input/test.txt 9 40
./sudoku_to_graph.exe input/test.txt graph.json
./cuda.exe input/test.txt output.txt measurements.txt 500 1000 0.05
```
