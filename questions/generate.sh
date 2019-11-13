id=$(hostname)
echo "running benchmark: read"
python3 plot/plot.py -b read -t 0.01 --savefig questions/read_$id.png
echo "running benchmark: array_sum"
python3 plot/plot.py -b array_sum -t 0.01 -s 512 609 724 4000000 --savefig questions/array_sum_$id.png
echo "running benchmark: dot_prod"
python3 plot/plot.py -b dot_prod -t 0.01 -s 512 609 724 4000000 --savefig questions/dot_prod_$id.png
echo "running benchmark: stride_sum"
python3 plot/plot.py -b stride_sum -f 6vec -t 0.01 -s 2 4 6 32 -s2 32 38 45 4000000 --savefig questions/stride_sum_1_$id.png
echo "running benchmark: stride_sum"
python3 plot/plot.py -b stride_sum -f 6vec -t 0.01 -s 2 4 8 128 -s2 32 38 45 4000000 --savefig questions/stride_sum_2_$id.png
echo "running benchmark: mat_sum"
python3 plot/plot.py -b mat_sum -s 6 12 18 1200 -s2 2 4 6 1200 -f 6vec -t 0.01 --savefig questions/mat_sum_$id.png
