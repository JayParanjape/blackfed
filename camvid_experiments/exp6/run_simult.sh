#!/bin/bash

cudalist=(4 5 6 7)
c=(1 2 3 4)
for i in 0 1 2 3; do
    python -u driver.py cuda:${cudalist[$i]} ${c[$i]} False False ../../data_configs/camvid.yml 1000 '' >> training_progress_c${c[$i]}.txt &
done