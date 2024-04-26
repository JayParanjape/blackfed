#!/bin/bash

cudalist=(2 3 4 5 6 7)
c=(1 2 3 4 5 6)
for i in 0 1 2 3 4 5; do
    python -u driver.py cuda:${cudalist[$i]} ${c[$i]} False False ../../data_configs/polypgen.yml 1000 '' >> training_progress_c${c[$i]}.txt &
done