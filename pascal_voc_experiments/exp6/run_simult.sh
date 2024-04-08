#!/bin/bash
cudalist=(0 7)
c=(9 10)
for i in 0 1; do
    python -u driver.py cuda:${cudalist[$i]} ${c[$i]} False False ../../data_configs/pascal_voc.yml 1000 '' >> training_progress_c${c[$i]}.txt &
done