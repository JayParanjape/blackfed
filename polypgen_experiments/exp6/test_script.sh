#!/bin/bash
# for i in 0 1 2 3 4 5; do
for i in 4; do
    echo "Testing for model $i \n"
    echo "Testing for model $i \n" >> testing_progress_4_all.txt

    for j in 1 2 3 4 5 6; do 
        python -u driver.py cuda:7 ${j} True False ../../data_configs/polypgen.yml 1000 saved_models3_dice/client_${i}_best_val.pth >> testing_progress_4_all.txt
    done
done