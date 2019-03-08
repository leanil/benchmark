#echo "time unit: microseconds"

# futhark-opencl baseline.fut
# echo sum
# futhark-dataset -b -g [10000000]f32 | ./baseline --entry-point sum -t /dev/stderr -r 10 > /dev/null
# echo inc
# futhark-dataset -b -g [10000000]f32 | ./baseline --entry-point inc -t /dev/stderr -r 10 > /dev/null
# echo prod
# futhark-dataset -b -g f32 -g [10000000]f32 | ./baseline --entry-point prod -t /dev/stderr -r 10 > /dev/null

# futhark-opencl 1d.fut
# echo dot
# futhark-dataset -b -g [10000000]f32 -g [10000000]f32 | ./1d --entry-point dot -t /dev/stderr -r 10 > /dev/null
# echo dot1
# futhark-dataset -b -g [10000000]f32 -g [10000000]f32 -g [10000000]f32 | ./1d --entry-point dot1 -t /dev/stderr -r 10 > /dev/null
# echo dot2
# futhark-dataset -b -g [10000000]f32 -g [10000000]f32 -g [10000000]f32 | ./1d --entry-point dot2 -t /dev/stderr -r 10 > /dev/null
# echo dot3
# futhark-dataset -b -g [10000000]f32 -g [10000000]f32 -g [10000000]f32 | ./1d --entry-point dot3 -t /dev/stderr -r 10 > /dev/null
# echo dot4
# futhark-dataset -b -g [10000000]f32 -g [10000000]f32 -g [10000000]f32 -g [10000000]f32 | ./1d --entry-point dot4 -t /dev/stderr -r 10 > /dev/null
# echo dot5
# futhark-dataset -b -g f32 -g [10000000]f32 -g f32 -g [10000000]f32 -g f32 -g [10000000]f32 -g f32 -g [10000000]f32 | ./1d --entry-point dot5 -t /dev/stderr -r 10 > /dev/null
# echo dot6
# futhark-dataset -b -g [10000000]f32 -g [10000000]f32 -g [10000000]f32 -g [10000000]f32 | ./1d --entry-point dot6 -t /dev/stderr -r 10 > /dev/null

i=4000
j=4000
k=4000
futhark-opencl 2d.fut
echo t1
futhark-dataset -b -g [$i][$j]f32 -g [$j]f32 | ./2d --entry-point t1 -t /dev/stderr -r 10 > /dev/null
echo t2
futhark-dataset -b -g [$i][$j]f32 -g [$j]f32 -g [$i]f32 | ./2d --entry-point t2 -t /dev/stderr -r 10 > /dev/null
echo t3
futhark-dataset -b -g [$i][$j]f32 -g [$i][$j]f32 -g [$j]f32 | ./2d --entry-point t3 -t /dev/stderr -r 10 > /dev/null
echo t4
futhark-dataset -b -g [$i][$j]f32 -g [$i][$j]f32 -g [$j]f32 -g [$j]f32 | ./2d --entry-point t4 -t /dev/stderr -r 10 > /dev/null
echo t5
futhark-dataset -b -g f32 -g [$i][$j]f32 -g f32 -g [$i][$j]f32 -g f32 -g [$j]f32 -g f32 -g [$j]f32 | ./2d --entry-point t5 -t /dev/stderr -r 10 > /dev/null
echo t6
futhark-dataset -b -g [$i]f32 -g [$j]f32 -g [$i]f32 -g [$j]f32 | ./2d --entry-point t6 -t /dev/stderr -r 10 > /dev/null
echo t7
futhark-dataset -b -g [$i][$j]f32 -g [$j][$k]f32 -g [$k]f32 | ./2d --entry-point t7 -t /dev/stderr -r 10 > /dev/null
echo t8
futhark-dataset -b -g [$i][$j]f32 -g [$i][$j]f32 -g [$j][$k]f32 -g [$k]f32 | ./2d --entry-point t8 -t /dev/stderr -r 10 > /dev/null
echo t9
futhark-dataset -b -g [$i][$k]f32 -g [$k][$j]f32 -g [$j]f32 -g [$j]f32 | ./2d --entry-point t9 -t /dev/stderr -r 10 > /dev/null
echo t10
futhark-dataset -b -g [$i][$k]f32 -g [$k][$j]f32 -g [$j][$k]f32 -g [$k]f32 | ./2d --entry-point t10 -t /dev/stderr -r 10 > /dev/null
