import os
import glob
from collections import defaultdict

input_folder = "output"
output_file = os.path.join(input_folder, "average.txt")

file_paths = [f for f in glob.glob(os.path.join(input_folder, "*")) if os.path.isfile(f)]

grouped_files = defaultdict(list)

for path in file_paths:
    filename = os.path.basename(path)
    if filename == "average.txt":
        continue
    prefix = filename
    if '.' in filename:
        prefix = filename.rsplit('.', 1)[0]
    grouped_files[prefix].append(path)

def parse_times_from_files(file_list):
    times = defaultdict(list)
    for filepath in file_list:
        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        size = int(parts[0])
                        time = float(parts[1])
                        times[size].append(time)
                    except ValueError:
                        continue
    return times

def compute_average(times_dict):
    return {size: sum(times) / len(times) for size, times in times_dict.items()}

with open(output_file, "w") as out:
    for prefix, files in grouped_files.items():
        out.write(f"Algorithm & parameters: {prefix}\n")
        times = parse_times_from_files(files)
        averages = compute_average(times)
        for size in sorted(averages.keys()):
            count = len(times[size])
            out.write(f"  Size {size}: average time = {averages[size]:.2f} s over {count} test(s)\n")
        out.write("\n")

print(f"Średnie czasy zapisane w {output_file}")
