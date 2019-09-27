from numpy import linspace

for i in range(8,2 ** 13,8):
    print(i)
for i in range(2**13, 2 ** 16, 64):
    print(i)
for i in linspace(16,25,1000):
    print(int(2 ** i))
