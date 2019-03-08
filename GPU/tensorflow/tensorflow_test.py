import baseline
import dim1
import dim2
import util

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tests = [#baseline.baseline_sum, baseline. baseline_inc, baseline.baseline_prod,
         #dim1.dot, dim1.dot1, dim1.dot2, dim1.dot3, dim1.dot4, dim1.dot5, dim1.dot6,
         dim2.t1, dim2.t2, dim2.t3, dim2.t4, dim2.t5, dim2.t6, dim2.t7, dim2.t8, dim2.t9, dim2.t10]
repeat = 5
if len(sys.argv) == 1:
    for i in range(len(tests)):
        print(tests[i].__name__)
        copy,calc = float("inf"),float("inf")
        for _ in range(repeat):
            a,b = util.nvprof(i)
            copy = min(copy, a)
            calc = min(calc,b)
        print("copy:",copy,"(s)")
        print("calc:",calc,"(s)")
else:
    print(tests[int(sys.argv[1])]())