import argparse
import collections
import csv
from datetime import datetime
import json
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import platform
import re
import subprocess
import sys
import time


def pin_to_core():
    if platform.system() == "Windows":
        return ["start", "/affinity", "0x4"]
    elif platform.system() == "Linux":
        return ["taskset", "0x4"]
    return None


def update_results(old_data, new_data):
    aggregate_logic = {
        "real_time": min,
        "cpu_time": min,
        "processing_speed": max,
        "iterations": max
    }
    for old, new in zip(old_data["benchmarks"], new_data["benchmarks"]):
        for key in old:
            if key in aggregate_logic:
                old[key] = aggregate_logic[key](old[key], new[key])
            else:
                assert old[key] == new[key]


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
    bench_args = pin_to_core() + [
        args.benchmark,
        "--benchmark_format=json"] + \
        ([f"--benchmark_filter={args.filter}"]
         if args.filter else []) + \
        args.extra
    if not os.path.exists(f"{base}/data"):
        os.mkdir(f"{base}/data")
    t0 = datetime.fromtimestamp(time.time())
    for i in range(args.repetitions):
        print(f"run #{i+1} / {args.repetitions}", end='\r')
        proc = subprocess.run(bench_args, input=' '.join(map(str, sizes)),
                              text=True, capture_output=True)
        if i == 0:
            t1 = datetime.fromtimestamp(time.time())
            end = t1 + (t1-t0)*(args.repetitions-1)
            print("ETA:", end.strftime("%H:%M:%S"))
            print("total length:", (t1-t0)*args.repetitions)
        try:
            result = json.loads(proc.stdout)
        except json.decoder.JSONDecodeError as ex:
            print(ex)
            print(proc)
            exit(1)
        if result["context"]["cpu_scaling_enabled"]:
            print("WARNING: CPU scaling is enabled.")
        if i == 0:
            data = result
        else:
            update_results(data, result)
    out_file = f"{base}/data/{t0.strftime('%Y-%m-%d_%H-%M-%S')}_" \
        f"{os.path.basename(args.benchmark)}_" \
        f"{result['context']['host_name']}.json"
    with open(out_file, 'w') as f:
        json.dump(data, f)
    return out_file


def use_log_scale(sizes):
    """use log2 scale on sizes, if it makes the gaps more even"""
    def gap_ratio(sizes):
        max_gap = min_gap = sizes[1]-sizes[0]
        for i in range(1, len(sizes)-1):
            max_gap = max(max_gap, sizes[i+1]-sizes[i])
            min_gap = min(min_gap, sizes[i+1]-sizes[i])
        return max_gap / min_gap
    log_sizes = [math.log(s, 2) for s in sizes]
    return gap_ratio(log_sizes) < gap_ratio(sizes)


def plot_data():
    result = json.load(open(args.plot))
    x_label = [x for x in result["benchmarks"]
               [0] if x.startswith("x_label")][0]
    cache_sizes = [cache["size"] for cache in result["context"]["caches"]
                   if cache["type"] != "Instruction"]
    Series = collections.namedtuple("Series", ['x', 'y'])
    data = {}
    for bench in result["benchmarks"]:
        name = bench["name"].rsplit('/', 1)[0]
        if args.filter and not re.search(args.filter, name):
            continue
        if not name in data:
            data[name] = Series([], [])
        data[name].x.append(int(bench[x_label]))
        data[name].y.append(float(bench["processing_speed"])/(1 << 30))
    sizes = list(data.values())[0].x
    fig, ax = plt.subplots()
    ax.set(xlabel=x_label.split(':')[1], title=os.path.basename(args.plot))
    if args.heatmap:
        y_label = list(data)[0].rsplit('/')[-2]
        data = sorted([(int(k.rsplit('/', 1)[1]), v) for k, v in data.items()])
        img = [row[1].y for row in data]
        hmap = ax.imshow(img, aspect='auto',
                         extent=(sizes[0], sizes[-1], data[-1][0], data[0][0]))
        for i, c in enumerate(cache_sizes, 1):
            ax.plot(sizes, [c / x for x in sizes], label=f"L{i} cache size")
        ax.set(ylabel=y_label,
               ylim=(data[-1][0], data[0][0]))
        fig.colorbar(hmap, ax=ax, orientation='horizontal',
                     label="data processing speed (GiB/s)")
    else:
        for name, values in data.items():
            ax.plot(values[0], values[1], "o-", label=name,
                    linewidth=0.5, markersize=2)
        if use_log_scale(sizes):
            ax.set_xscale("log", basex=2)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set(ylabel="data processing speed (GiB/s)")
        for c in cache_sizes:
            if sizes[0] <= c <= sizes[-1]:
                ax.axvline(c, color='black', dashes=[2, 10])
    ax.legend()
    fig.set_size_inches(18.53, 9.55)
    if args.savefig == None:
        plt.show()
    elif args.savefig == "":
        fig.savefig(os.path.splitext(args.plot)[0]+".png")
    elif os.path.dirname(args.savefig):
        fig.savefig(args.savefig)
    else:
        fig.savefig(f"{base}/data/{args.savefig}")


base = os.path.dirname(os.path.abspath(__file__))
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
parser.add_argument("--heatmap", action='store_true',
                    help="plot performance as a heatmap of 2 parameters")
parser.add_argument("-p", "--plot", help="just plot the given data file")
parser.add_argument("-f", "--filter", help="benchmark tasks to run (regex)")
parser.add_argument("-r", "--repetitions", type=int, default=3,
                    help="repeat measurements multiple times to reduce noise (default: 3)")
parser.add_argument("--savefig", nargs='?', const="",
                    help="save the plot with the given name, instead of showing")
parser.add_argument("extra", nargs=argparse.REMAINDER,
                    help="extra args passed to google benchmark")

args = parser.parse_args()
if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()
if args.extra and args.extra[0] == "--":
    args.extra.pop(0)
if not args.plot:
    args.plot = collect_data()
if not args.collect:
    plot_data()
