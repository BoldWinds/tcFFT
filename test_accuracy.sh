#!/bin/bash

if [ -d "build" ]; then
    rm -rf build/*
else
    mkdir build
fi

make

touch build/test_accuracy.txt

# 半精度误差测试
echo "Test for half precision:" >> build/test_accuracy.txt
for ((i = 1; i <= 100; i++)); do
    random_number=$((RANDOM % 100 + 1))
    echo "test $i" >> build/test_accuracy.txt
    ./build/test -b 1 -n 256 -p 0 -s $random_number -t 1 >> build/test_accuracy.txt
    echo "" >> build/test_accuracy.txt
done

# 单精度误差测试
#echo "Test for single precision:" >> build/test_accuracy.txt
#for ((i = 1; i <= 10; i++)); do
#    random_number=$((RANDOM % 100 + 1))
#    echo "test $i" >> build/test_accuracy.txt
#    ./build/test -b 1 -n 256 -p 1 -s $random_number -t 1 >> build/test_accuracy.txt
#    echo "" >> build/test_accuracy.txt
#done
