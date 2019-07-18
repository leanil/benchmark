import csv
import subprocess
import time

def nvprof(i, j):
    nvprof_args = ["nvprof", "--unified-memory-profiling", "off", "--print-gpu-summary", "--csv", "./cuBLAS_test"]
    time_units = {"s":1, "ms":1e-3, "us":1e-6}
    out = subprocess.check_output(nvprof_args + [str(i),str(j)], stderr=subprocess.STDOUT, encoding="utf-8")
    parser = csv.reader(out.splitlines())
    calc, copy = 0,0
    for row in parser:
        if row[0] == "GPU activities":
            if row[7].find("memcpy") != -1:
                copy += float(row[2])
            else:
                calc += float(row[2])
        elif len(row) > 2 and row[2] in time_units:
           unit = time_units[row[2]]
    return(copy * unit, calc * unit)

tests = [["baseline_sum", "baseline_inc", "baseline_prod"], ["dot", "dot2", "dot3", "dot4", "dot5"], ["t1", "t3", "t4", "t5", "t7", "t8", "t9", "t10"]]
repeat = 5
for i in range(len(tests)):
    for j in range(len(tests[i])):
        print(tests[i][j])
        copy,calc = float("inf"),float("inf")
        for _ in range(repeat):
            a,b = nvprof(i,j)
            copy = min(copy, a)
            calc = min(calc,b)
        print("copy:",copy,"(s)")
        print("calc:",calc,"(s)")