#!/bin/bash
for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17; do
    echo "Testing for model $i \n"
    echo "Testing for model $i \n"

    for j in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18; do        
        python -u driver.py cuda:1 ${j} True False ../../data_configs/cityscapes.yml 1000 saved_models3_dice/client_${i}_best_val.pth
    done
done