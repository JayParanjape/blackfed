#!/bin/bash
for i in 0 1 2 3; do
    echo "Testing for model $i \n"
    echo "Testing for model $i \n" >> testing_progress.txt

    for j in 1 2 3 4; do        
        python -u driver.py cuda:4 ${j} True False ../../data_configs/camvid.yml 1000 saved_models/client_${i}_best_val.pth >> testing_progress.txt
    done
done