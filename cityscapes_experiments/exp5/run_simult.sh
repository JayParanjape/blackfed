#!/bin/bash
cudalist=(8 9)
c=(17 18)
for i in 0 1; do
    python -u driver.py cuda:${cudalist[$i]} ${c[$i]} False False ../../data_configs/cityscapes.yml 1000 '' >> training_progress_c${c[$i]}.txt &
done