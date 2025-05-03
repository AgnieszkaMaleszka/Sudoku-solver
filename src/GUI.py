import tkinter as tk
import subprocess
import os
import random
import time
import colorsys
import threading

PASTEL_BLUE = "#dceeff"
NAVY = "#003366"
WHITE = "#ffffff"

class SudokuGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sudoku Solver GUI")
        self.root.configure(bg=PASTEL_BLUE)
        self.root.geometry("1150x880")

        self.grid_size = tk.IntVar(value=9)
        self.empty_cells = tk.IntVar(value=40)
        self.population = tk.IntVar(value=100)
        self.iterations = tk.IntVar(value=500)
        self.mutation_chance = tk.DoubleVar(value=0.05)
        self.executable = tk.StringVar(value="../build/Release/seq.exe")

        self.output_file = "../output/output.txt"
        self.measurement_file = "../output/measurements.txt"
        self.generated_input = "../input/test.txt"
        self.solver_process = None
        self.spinner_running = False

        self.board_entries = []
        self.status_text = tk.StringVar(value="Gotowe.")
        self.empty_cell_warning = tk.StringVar(value="")
        self.mutation_warning = tk.StringVar(value="")

        self.build_interface()

    def build_interface(self):
        main_frame = tk.Frame(self.root, bg=PASTEL_BLUE)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        side_frame = tk.LabelFrame(main_frame, text="Parametry", padx=10, pady=10, bg=PASTEL_BLUE, fg=NAVY, font=("Arial", 11, "bold"))
        side_frame.pack(side="right", fill="y", padx=10)

        tk.Label(side_frame, text="Rozmiar planszy:", bg=PASTEL_BLUE, fg=NAVY).pack(anchor="w")
        tk.Radiobutton(side_frame, text="9x9", variable=self.grid_size, value=9, command=self.build_board, bg=PASTEL_BLUE).pack(anchor="w")
        tk.Radiobutton(side_frame, text="16x16", variable=self.grid_size, value=16, command=self.build_board, bg=PASTEL_BLUE).pack(anchor="w")

        self.add_labeled_entry(side_frame, "Puste komórki:", self.empty_cells, self.empty_cell_warning, validate_type="empty")
        tk.Label(side_frame, textvariable=self.empty_cell_warning, fg="red", bg=PASTEL_BLUE, wraplength=180).pack()

        self.add_labeled_entry(side_frame, "Populacja:", self.population)
        self.add_labeled_entry(side_frame, "Iteracje:", self.iterations)
        self.add_labeled_entry(side_frame, "Mutacja (0-1):", self.mutation_chance, self.mutation_warning, validate_type="mutation")
        tk.Label(side_frame, textvariable=self.mutation_warning, fg="red", bg=PASTEL_BLUE, wraplength=180).pack()

        tk.Label(side_frame, text="Algorytm:", bg=PASTEL_BLUE, fg=NAVY).pack(anchor="w", pady=5)
        tk.OptionMenu(side_frame, self.executable, "../build/Release/seq.exe", "../build/Release/cuda.exe", "../build/Release/omp.exe").pack(fill="x")

        def style_btn(widget): widget.configure(bg=NAVY, fg="white", activebackground="#002244", activeforeground="white")

        b1 = tk.Button(side_frame, text="Zaktualizuj planszę", command=self.build_board)
        b2 = tk.Button(side_frame, text="Rozwiąż Sudoku", command=self.run_solver)
        b3 = tk.Button(side_frame, text="Wczytaj losowy test", command=self.run_random_test_and_load)
        b4 = tk.Button(side_frame, text="Przerwij działanie", command=self.terminate_solver)

        for b in [b1, b2, b3, b4]:
            style_btn(b)
            b.pack(fill="x", pady=5)

        self.time_label = tk.Label(side_frame, text="Czas: -", font=("Arial", 10, "italic"), bg=PASTEL_BLUE, fg=NAVY)
        self.time_label.pack(pady=10)

        self.status_label = tk.Label(side_frame, textvariable=self.status_text, wraplength=200, fg="gray", bg=PASTEL_BLUE)
        self.status_label.pack(pady=10)

        center_frame = tk.Frame(main_frame, bg=PASTEL_BLUE)
        center_frame.pack(expand=True)
        self.board_container = tk.Frame(center_frame, bg=PASTEL_BLUE)
        self.board_container.pack()
        self.build_board()

    def add_labeled_entry(self, parent, label, variable, warning_var=None, validate_type=None):
        tk.Label(parent, text=label, bg=PASTEL_BLUE, fg=NAVY).pack(anchor="w")
        entry = tk.Entry(parent, textvariable=variable, width=10)
        entry.pack(fill="x", pady=2)

        def validate(*args):
            try:
                val = float(variable.get())
                size = max(9, self.grid_size.get())
                if validate_type == "empty":
                    max_allowed = size * size - 1
                    if val > max_allowed:
                        warning_var.set(f"Maksymalna liczba pustych: {max_allowed}")
                    else:
                        warning_var.set("")
                elif validate_type == "mutation":
                    if val < 0 or val > 1:
                        warning_var.set("Mutacja musi być między 0 a 1.")
                    else:
                        warning_var.set("")
            except:
                if warning_var:
                    warning_var.set("Nieprawidłowa wartość.")

        if validate_type:
            variable.trace_add("write", lambda *args: validate())

    def run_spinner(self):
        spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        while self.spinner_running:
            self.status_text.set(f"{spinner[i % len(spinner)]} Rozwiązywanie Sudoku...")
            i += 1
            time.sleep(0.1)

    def run_solver(self):
        if self.population.get() <= 0 or self.iterations.get() <= 0:
            self.status_text.set("Populacja i iteracje muszą być większe od zera.")
            return
        if self.mutation_chance.get() < 0 or self.mutation_chance.get() > 1:
            self.status_text.set("Mutacja musi być między 0 a 1.")
            return
        if not os.path.exists(self.executable.get()):
            self.status_text.set(f"Nie znaleziono pliku: {self.executable.get()}")
            return

        temp_input = "../input/temp_input.txt"
        self.save_board_to_file(temp_input)
        cmd = [
            self.executable.get(),
            temp_input,
            self.output_file,
            self.measurement_file,
            str(self.population.get()),
            str(self.iterations.get()),
            str(self.mutation_chance.get())
        ]

        self.spinner_running = True
        threading.Thread(target=self.run_spinner, daemon=True).start()

        def run_process():
            try:
                start = time.time()
                self.solver_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                stdout, stderr = self.solver_process.communicate()
                elapsed = time.time() - start
                self.spinner_running = False
                for _ in range(10):
                    if os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0:
                        break
                    time.sleep(0.1)

                if os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0:
                    self.root.after(0, lambda: self.load_board_from_file(self.output_file))
                    self.root.after(0, lambda: self.time_label.config(text=f"Czas: {elapsed:.2f} s"))
                    self.root.after(0, lambda: self.status_text.set("✅ Sudoku rozwiązane."))
                else:
                    self.root.after(0, lambda: self.status_text.set("❌ Solver zakończył się, ale plik wynikowy jest pusty."))
            except Exception as e:
                self.spinner_running = False
                self.root.after(0, lambda: self.status_text.set(f"Błąd: {e}"))

        threading.Thread(target=run_process).start()

    def save_board_to_file(self, filename):
        with open(filename, 'w') as f:
            for row in self.board_entries:
                line = ' '.join(e.get() if e.get() else '0' for e in row)
                f.write(line + '\n')

    def load_board_from_file(self, filename):
        if not os.path.exists(filename):
            self.status_text.set(f"Plik {filename} nie istnieje.")
            return
        with open(filename, 'r') as f:
            lines = f.readlines()
        self.grid_size.set(len(lines))
        self.build_board()
        for i, line in enumerate(lines):
            values = line.strip().split()
            for j, val in enumerate(values):
                self.board_entries[i][j].delete(0, tk.END)
                if val != '0':
                    self.board_entries[i][j].insert(0, val)
        self.color_board()

    def build_board(self):
        for widget in self.board_container.winfo_children():
            widget.destroy()
        self.board_entries = []
        size = max(9, self.grid_size.get())
        square_size = max(20, 540 // size)
        block = int(size ** 0.5)

        def validate_input(P):
            return P == "" or (P.isdigit() and 1 <= int(P) <= size)

        vcmd = (self.root.register(validate_input), '%P')

        for i in range(size):
            row = []
            for j in range(size):
                e = tk.Entry(self.board_container, width=2, font=('Courier', int(square_size / 1.8)), justify='center',
                             bg=WHITE, relief='solid', bd=1, validate="key", validatecommand=vcmd)
                e.grid(row=i, column=j, ipadx=square_size // 4, ipady=square_size // 8,
                       padx=(3 if j % block == 0 else 1, 3 if (j + 1) % block == 0 else 1),
                       pady=(3 if i % block == 0 else 1, 3 if (i + 1) % block == 0 else 1),
                       sticky='nsew')
                e.bind("<KeyRelease>", lambda event: self.color_board())
                row.append(e)
            self.board_container.grid_rowconfigure(i, weight=1)
            self.board_entries.append(row)
        for j in range(size):
            self.board_container.grid_columnconfigure(j, weight=1)
        self.color_board()

    def color_board(self):
        value_colors = {}
        next_color = 0
        for row in self.board_entries:
            for cell in row:
                val = cell.get()
                if val and val != '0':
                    if val not in value_colors:
                        hue = (next_color * 37) % 360
                        r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 0.3, 0.85)
                        value_colors[val] = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
                        next_color += 1
                    cell.configure(bg=value_colors[val])
                else:
                    cell.configure(bg=WHITE)

    def run_random_test_and_load(self):
        size = max(9, self.grid_size.get())
        max_allowed = size * size - 1
        requested = self.empty_cells.get()
        if requested > max_allowed:
            self.empty_cell_warning.set(f"Maksymalna liczba pustych: {max_allowed}")
            return
        else:
            self.empty_cell_warning.set("")

        seed = random.randint(0, 99999)
        output_file = self.generated_input
        cmd = [
            "../build/Release/sudoku_generator.exe",
            output_file,
            str(size),
            str(requested),
            str(seed)
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and os.path.exists(output_file):
                self.load_board_from_file(output_file)
                self.status_text.set(f"Wczytano planszę z pliku: {output_file}")
            else:
                self.status_text.set("Błąd: nie udało się wygenerować planszy.")
        except Exception as e:
            self.status_text.set(f"Błąd generatora:{e}")

    def terminate_solver(self):
        if self.solver_process and self.solver_process.poll() is None:
            self.solver_process.terminate()
            self.status_text.set("Solver został przerwany.")
        else:
            self.status_text.set("Brak aktywnego solvera do zatrzymania.")

if __name__ == "__main__":
    root = tk.Tk()
    app = SudokuGUI(root)
    root.mainloop()
