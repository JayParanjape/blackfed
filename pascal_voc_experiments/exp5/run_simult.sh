#!/bin/bash
cudalist=(0 1 2 3 4 5 6 7)
c=(1 2 3 4 5 6 7 8)
for i in 0 1 2 3 4 5 6 7; do
    python -u driver.py cuda:${cudalist[$i]} ${c[$i]} False False ../../data_configs/pascal_voc.yml 1000 '' >> training_progress_c${c[$i]}.txt &
done