import argparse
import csv
from datetime import datetime
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import os
import platform
import subprocess
import time


def pin_to_core():
    if platform.system() == "Windows":
        return ["start", "/affinity", "0x4"]
    elif platform.system() == "Linux":
        return ["taskset", "0x4"]
    return None


def collect_data():
    if args.step_add != 0 or args.step_mul != 1:
        sizes = []
        s = args.min_size
        while s <= args.max_size:
            sizes.append(s)
            s = args.step_mul * s + args.step_add
    else:
        with open(args.sizes) as size_file:
            sizes = [int(s) for s in size_file.readlines()
                     if args.min_size <= int(s) <= args.max_size]
    time_stamp = datetime.fromtimestamp(
        time.time()).strftime('%H-%M-%S_%Y-%m-%d')
    out_file = f"{base}/data/{time_stamp}.txt"
    bench_args = pin_to_core() + [
        args.benchmark,
        f"--benchmark_repetitions={args.repetitions}",
        "--benchmark_report_aggregates_only=true",
        "--benchmark_out=" + out_file,
        "--benchmark_out_format=csv"] + \
        ([f"--benchmark_filter={args.filter}"]
         if args.filter else []) + \
        args.extra
    if not os.path.exists("data"):
        os.mkdir("data")
    subprocess.run(bench_args, input=' '.join(map(str, sizes)), text=True)
    return out_file

# use log2 scale on sizes, if it makes the gaps more even


def use_log_scale(sizes):
    def gap_ratio(sizes):
        gaps = []
        for i in range(len(sizes)-1):
            gaps.append(sizes[i+1]-sizes[i])
        return max(gaps) / min(gaps)
    log_sizes = [math.log(s, 2) for s in sizes]
    return gap_ratio(log_sizes) < gap_ratio(sizes)


def get_cache_sizes():
    if platform.system() == "Linux":
        sizes = subprocess.check_output(
            "lscpu | awk ' /'cache'/ {print $3} '", shell=True, text=True).splitlines()
        sizes.pop(1)
        return [int(s[:-1])*1024 for s in sizes]
    return None


def plot_data():
    reader = csv.reader(args.plot)
    data = defaultdict(list)
    sizes = []
    for row in reader:
        if row[0].endswith("max"):
            name, size = row[0].rsplit('/', 1)
            data[name].append(float(row[10])/(1 << 30))
            if len(data) == 1:
                sizes.append(int(size[:-4]))
    for name, y in data.items():
        plt.plot(sizes, y, label=name)
    if use_log_scale(sizes):
        plt.xscale("log", basex=2)
    plt.xlabel('data size (Bytes)')
    plt.ylabel('data processing speed (GiB/s)')
    for s in get_cache_sizes():
        if sizes[0] <= s <= sizes[-1]:
            plt.axvline(s, color='black', dashes=[2, 10])
    plt.legend()
    plt.show()


base = os.path.dirname(__file__)
parser = argparse.ArgumentParser(
    description="Run the specified benchmark and plot the result.")
parser.add_argument("-b", "--benchmark",
                    help="path to the benchmark executable")
parser.add_argument("--sizes", default=f"{base}/sizes.txt",
                    help="file containing benchmark sizes (1 integer per row)")
parser.add_argument("-s", "--min_size", type=int, default=1,
                    help="minimum benchmark size (bytes)")
parser.add_argument("-S", "--max_size", type=int, default=1 << 27,
                    help="maximum benchmark size (bytes)")
parser.add_argument("--step_add", type=int, default=0,
                    help="additive part of size step (overrides SIZES)")
parser.add_argument("--step_mul", type=int, default=1,
                    help="multiplicative part of size step (overrides SIZES)")
parser.add_argument("-c", "--collect", action='store_true',
                    help="just collect data (don't plot it)")
parser.add_argument("-p", "--plot", type=open,
                    help="just plot the given data file")
parser.add_argument("-f", "--filter", help="benchmark tasks to run (regex)")
parser.add_argument("-r", "--repetitions", type=int, default=3,
                    help="repeat measurements multiple times to reduce noise (default: 3)")
parser.add_argument("extra", nargs=argparse.REMAINDER,
                    help="extra args passed to google benchmark")

args = parser.parse_args()
if args.extra and args.extra[0] == "--":
    args.extra.pop(0)
if not args.plot:
    args.plot = open(collect_data())
if not args.collect:
    plot_data()
