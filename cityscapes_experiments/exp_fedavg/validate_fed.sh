#!/bin/bash
echo "............EVALUATING FED MODEL........."
for c in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18; do
    python -u ../../train_baselines.py cuda:6 $c True False ../../data_configs/cityscapes.yml 10 fed_learning_model.pth >> fed_training_progress.txt
done 
echo -e "\nNext epoch\n" >> fed_training_progress.txt
