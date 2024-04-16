#!/bin/bash
for i in 0 1 2 3 4 5 6 7 8 9 10; do
    echo "Testing for model $i \n"
    echo "Testing for model $i \n" >> testing_progress.txt

    for j in 1 2 3 4 5 6 7 8 9 10; do        
        python -u driver.py cuda:7 ${j} True False ../../data_configs/pascal_voc.yml 1000 saved_models/client_${i}_best_val.pth >> testing_progress_iou.txt
    done
done