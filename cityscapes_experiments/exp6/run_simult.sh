#!/bin/bash
# cudalist=(0 1 2 3 4 5 6 7)
# c=(1 2 3 4 5 6 7 8)
# for i in 0 1 2 3 4 5 6 7; do
#     python -u driver.py cuda:${cudalist[$i]} ${c[$i]} False False ../../data_configs/cityscapes.yml 1000 '' >> training_progress_c${c[$i]}.txt &
# done

cudalist=(0 1 2 3 4 5 6 7)
c=(9 10 11 12 13 14 15 16)
for i in 0 1 2 3 4 5 6 7; do
    python -u driver.py cuda:${cudalist[$i]} ${c[$i]} False False ../../data_configs/cityscapes.yml 1000 '' >> training_progress_c${c[$i]}.txt &
done

# cudalist=(5 7)
# c=(17 18)
# for i in 0 1; do
#     python -u driver.py cuda:${cudalist[$i]} ${c[$i]} False False ../../data_configs/cityscapes.yml 1000 '' >> training_progress_c${c[$i]}.txt &
# done