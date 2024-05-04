#!/bin/bash

cudalist=(7 8 9)
c=(1 2 3)
for i in 0 1 2 ; do
    python -u driver.py cuda:${cudalist[$i]} ${c[$i]} False False ../../data_configs/camvid.yml 1000 '' >> training_progress_c${c[$i]}.txt &
done