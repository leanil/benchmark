import argparse
import collections
import csv
from datetime import datetime
import json
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
import os
import platform
import png
import re
import subprocess
import sys
import time


def pin_to_core():
    if platform.system() == "Windows":
        return ["start", "/affinity", "0x4"]
    elif platform.system() == "Linux":
        # lscpu -p=CACHE | awk -F: '/^[^#]/ { print $1 }'
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

def get_string_id(context):
    return f'{context["date"].replace(" ", "_")}_{context["benchmark_name"]}_{context["host_name"]}'

def get_sizes(sizes):
    if len(sizes) == 1:
        with open(sizes[0]) as size_file:
            return [int(s) for s in size_file.readlines()]
    else:
        sizes = [int(i) for i in sizes]
        mul = (sizes[2] - sizes[1]) / (sizes[1]-sizes[0])
        add = sizes[1] - sizes[0]*mul
        result, i = [], sizes[0]
        while i <= sizes[3]:
            result.append(int(i))
            i = i * mul + add
        return result


def collect_data():
    sizes = ' '.join(map(str, get_sizes(args.sizes)))
    if conf["params"] == 2 or args.sizes2:
        s2 = args.sizes2 if args.sizes2 else args.sizes
        sizes += '\n' + ' '.join(map(str, get_sizes(s2)))
    bench_args = pin_to_core() + [
        exe_path,
        "--benchmark_format=json",
        f"--benchmark_min_time={args.min_time}",
        f"--benchmark_filter={args.filter}"
    ] + args.extra
    if not os.path.exists(f"{base}/data"):
        os.mkdir(f"{base}/data")
    t0 = datetime.fromtimestamp(time.time())
    for i in range(args.repetitions):
        print(f"run #{i+1} / {args.repetitions}", end='\r')
        proc = subprocess.run(bench_args, input=sizes,
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
    data["context"]["benchmark_name"] = args.benchmark
    out_file = f'{base}/data/{get_string_id(data["context"])}.json'
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

def get_json(filename):
    if(os.path.splitext(filename)[1] == ".png"):
        chunks = list(png.Reader(filename).chunks())
        return json.loads(chunks[-2][1])
    return json.load(open(filename))

def append_json_to_png(filename, data):
    chunks = list(png.Reader(filename).chunks())
    chunks.insert(len(chunks)-1,(b"tEXt",bytes(json.dumps(data),"utf-8")))
    with open(filename, 'wb') as file:
        png.write_chunks(file, chunks)

def plot_data():
    result = get_json(args.plot)
    cache_sizes = [cache["size"] // 1000 * 1024 for cache in result["context"]["caches"]
                   if cache["type"] != "Instruction"]
    Series = collections.namedtuple("Series", ['x', 'y'])
    data = {}
    for bench in result["benchmarks"]:
        if args.filter and not re.search(args.filter, bench["name"]):
            continue
        name, pos = bench["name"].rsplit('/', 1)
        if not name in data:
            data[name] = Series([], [])
        data[name].x.append(int(pos))
        data[name].y.append(float(bench["processing_speed"])/(1 << 30))
    sizes = list(data.values())[0].x
    fig, ax = plt.subplots()
    ax.set(xlabel=conf["axes"]["x"], ylabel=conf["axes"]["y"],
           title=get_string_id(result["context"]))
    if conf["params"] == 2:
        data = sorted([(int(k.rsplit('/', 1)[1]), v) for k, v in data.items()])
        img = [row[1].y for row in data]
        if max(map(max, img)) > args.heatmap_max:
            print("Max value exceeds color range!")
        colors = ["#000000", "#00FF00", "#008844", "#00FFFF", "#004488", "#0000FF", "#440088",
                  "#FF00FF", "#880044", "#FF0000", "#884400", "#FFFF00", "#444444", "#FFFFFF", "#000000"]
        cmap = LinearSegmentedColormap.from_list('my_colormap', colors, 2048)
        hmap = ax.imshow(img, aspect='auto', vmin=0, vmax=args.heatmap_max, cmap=cmap,
                         extent=(sizes[0], sizes[-1], data[-1][0], data[0][0]))
        for i, c in enumerate(cache_sizes, 1):
            ax.plot(sizes, [c / x / conf["read_factor"]
                            for x in sizes], label=f"L{i} cache size")
        ax.set(ylim=(data[-1][0], data[0][0]))
        fig.colorbar(hmap, ax=ax, orientation='horizontal',
                     label=conf["axes"]["z"])
    else:
        for name, values in data.items():
            ax.plot(values[0], values[1], "o-", label=name,
                    linewidth=0.5, markersize=2)
        if use_log_scale(sizes):
            ax.set_xscale("log", basex=2)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for c in cache_sizes:
            c /= conf["read_factor"]
            if sizes[0] <= c <= sizes[-1]:
                ax.axvline(c, color='black', dashes=[2, 10])
    ax.legend()
    fig.set_size_inches(18.53, 9.55)
    if args.savefig is None:
        plt.show()
    else:
        if args.savefig == "":
            filename = os.path.splitext(args.plot)[0]+".png"
        elif os.path.dirname(args.savefig):
            filename = args.savefig
        else:
            filename = f"{base}/data/{args.savefig}"
        fig.savefig(filename)
        append_json_to_png(filename, result)


base = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(
    description="Run the specified benchmark and plot the result.")
parser.add_argument("-b", "--benchmark",
                    help="name of benchmark to run")
parser.add_argument("-s", "--sizes", nargs='+', default=[f"{base}/sizes.txt"],
                    help="""4 integers <start step1 step2 stop> specifying a sequence,
                            or path to a file containing benchmark sizes (1 integer per row)""")
parser.add_argument("-s2", "--sizes2", nargs='+',
                    help="sizes of second dimension (see --sizes)")
parser.add_argument("-c", "--collect", action='store_true',
                    help="just collect data (don't plot it)")
parser.add_argument("-p", "--plot", metavar="FILE",
                    help="just plot the given data file")
parser.add_argument("--heatmap_max", type=float, default=140,
                    help="the end of the heatmap color scale")
parser.add_argument("-f", "--filter", metavar="REGEX", default="",
                    help="benchmark tasks to run")
parser.add_argument("-t", "--min_time", type=float, default=0.1,
                    help="run each benchmark for MIN_TIME seconds")
parser.add_argument("-r", "--repetitions", type=int, default=3,
                    help="repeat measurements multiple times to reduce noise (default: 3)")
parser.add_argument("--savefig", nargs='?', const="", metavar="FILE",
                    help="save the plot with the given name, instead of showing")
parser.add_argument("extra", nargs=argparse.REMAINDER,
                    help="extra args passed to google benchmark")

args = parser.parse_args()
if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()
if args.extra and args.extra[0] == "--":
    args.extra.pop(0)
conf = json.load(open(f"{base}/config.json"))
if not args.benchmark in conf:
    print(f"unknown benchmark: {args.benchmark}")
    sys.exit()
exe_path = f"{conf['build_dir']}/app/{args.benchmark}/{args.benchmark}"
conf = conf[args.benchmark]
if not args.plot:
    args.plot = collect_data()
if not args.collect:
    plot_data()
