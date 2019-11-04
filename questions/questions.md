# read

* Is this a correct way to measure the sustained bandwidth of cache levels?
* (Why are there increasing segments (on AMD)?)

# array_sum

* (Why is mov + add not the same as just add (on AMD)?)

# dot_prod

* Why is it faster than array_sum in the L3 region?

# stride_sum

`python3 plot/plot.py -b stride_sum -p plot/data/2019-11-04_07-48-22_stride_sum_RED-FLASH.json -f"6vec/14|6vec/18|6vec/20|6vec/22|6vec/24|6vec/26|6vec/28"`
* Why do the curves with strides larger than 22 stay the same after the L2 boundary?
* (Why are the L3 throughput levels at 8,16 stride not the same as at 32,64,128 stride?)

`python3 plot/plot.py -b stride_sum -f 6vec -t 0.01 -s 2 4 8 128 -s2 32 38 45 4000000`

* Any stride larger than 8 and not a multiple of 16 should look like 8, correct?


# mat_sum

`python3 plot/plot.py -b mat_sum -s 6 12 18 1200 -s2 2 4 6 1200 -f 6vec -t 0.01`

* light blue: data is stored in L3, current column in L2. Is it bound by L3 -> L2 or L2 -> L1 BW?
* Why isn't the left part of light blue (that should fit in L2) different?
    - Because it's already limited by L2 -> L1? Then what are the green columns?
* green: L3 -> L2 or ram -> L3 BW?
* blue: data in L3, current column in L1. How to calculate BW?
* red: data in L2, current column in L1. Is this L2 -> L1 BW?
    - What are the yellow columns?
* What are pink areas?


# Performance counters

* How to detect bank conflicts?
* How to verify that the program is compute bound?
* (LsMabAlloc)