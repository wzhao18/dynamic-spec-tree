#!/bin/bash

temps_list=()
p_list=()

# create elements between 0.2 to 0.8 
step=$(echo "1 / 10" | bc -l)

for i in {2..8}
do
  value=$(echo "$i * $step" | bc -l)
  temps_list+=($value)
done

# create elements between 0.2 to 0.8 
step=$(echo "1 / 10" | bc -l)
for i in {2..10}
do
  value=$(echo "$i * $step" | bc -l)
  p_list+=($value)
done


for temp in "${temp_list[@]}"
do 
    for p in "${p_list[@]}"
    do  
        python3 testbed_dynamic.py --T $temp --draft_T 0.6 --P $p --M 512 > dynamic.log
    done
done
    