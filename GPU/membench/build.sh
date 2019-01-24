g++-7 -I/usr/include/ -L/usr/lib/x86_64-linux-gnu/ clMem.cpp -O3 -o m.out -lOpenCL
nvcc ./cuMem.cu -o c.out -O3
