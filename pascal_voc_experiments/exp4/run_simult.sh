#!/bin/bash
cudalist=(1 2 4 7)
c=(7 8 9 10)
for i in 0 1 2 3; do
    python -u driver.py cuda:${cudalist[$i]} ${c[$i]} False False ../../data_configs/pascal_voc.yml 1000 '' >> training_progress_c${c[$i]}.txt &
done