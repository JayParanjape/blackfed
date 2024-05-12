#!/bin/bash
for j in 1 2 3 4; do        
    python -u driver.py cuda:4 $j True False ../../data_configs/camvid.yml 1000 saved_models_dice/client_super_best_val.pth >> testing_progress.txt
done