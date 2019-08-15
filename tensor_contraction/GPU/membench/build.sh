g++-7 -I/usr/include/ -L/usr/lib/x86_64-linux-gnu/ clMem.cpp -O3 -o clMem -lOpenCL
nvcc ./cuMem.cu -o cuMem -O3
