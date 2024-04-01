#!/bin/bash
for c in 1 2 3; do
    python -u ../../train_baselines.py cuda:0 $c True False ../../data_configs/isic.yml 10 fed_learning_model.pth >> fed_training_progress.txt
done 
echo -e "\nNext epoch\n" >> fed_training_progress.txt
