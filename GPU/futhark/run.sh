futhark-opencl baseline.fut
echo sum
futhark-dataset -b -g [10000000]f32 | ./baseline --entry-point sum -t /dev/stderr -r 10 > /dev/null
echo prod
futhark-dataset -b -g f32 -g [10000000]f32 | ./baseline --entry-point prod -t /dev/stderr -r 10 > /dev/null

futhark-opencl 1d.fut
echo dot
futhark-dataset -b -g [10000000]f32 -g [10000000]f32 | ./1d --entry-point dot -t /dev/stderr -r 10 > /dev/null
echo dot1
futhark-dataset -b -g [10000000]f32 -g [10000000]f32 -g [10000000]f32 | ./1d --entry-point dot1 -t /dev/stderr -r 10 > /dev/null
echo dot2
futhark-dataset -b -g [10000000]f32 -g [10000000]f32 -g [10000000]f32 | ./1d --entry-point dot2 -t /dev/stderr -r 10 > /dev/null
echo dot3
futhark-dataset -b -g [10000000]f32 -g [10000000]f32 -g [10000000]f32 | ./1d --entry-point dot3 -t /dev/stderr -r 10 > /dev/null
echo dot4
futhark-dataset -b -g [10000000]f32 -g [10000000]f32 -g [10000000]f32 -g [10000000]f32 | ./1d --entry-point dot4 -t /dev/stderr -r 10 > /dev/null
echo dot5
futhark-dataset -b -g f32 -g [10000000]f32 -g f32 -g [10000000]f32 -g f32 -g [10000000]f32 -g f32 -g [10000000]f32 | ./1d --entry-point dot5 -t /dev/stderr -r 10 > /dev/null
echo dot6
futhark-dataset -b -g [10000000]f32 -g [10000000]f32 -g [10000000]f32 -g [10000000]f32 | ./1d --entry-point dot6 -t /dev/stderr -r 10 > /dev/null