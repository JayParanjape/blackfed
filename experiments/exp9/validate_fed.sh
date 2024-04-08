#!/bin/bash
for c in 1 2 3 4 5 6; do
    python -u ../../train_baselines.py cuda:0 $c True False ../../data_configs/polypgen.yml 10 fed_learning_model.pth >> fed_training_progress.txt
done 
echo -e "\nNext epoch\n" >> fed_training_progress.txt