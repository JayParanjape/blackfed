#!/bin/bash
echo "FedAvg Algorithm for CAMVid Dataset"
num_fed_epochs=10
num_center_epochs=25
cudalist=(0 1 2 3)

#initial training
# for c in 1 2 3 4; do
#         python ../../train_baselines.py cuda:${cudalist[$c-1]} $c "False" "False" "../../data_configs/camvid.yml" $num_center_epochs "." &
#     done
# #wait for all centers to train
# wait

for i in $(seq 1 $num_fed_epochs); do
    #step 1 - perform fedavg
    python ../../fed_avg.py "fedavg"
    #step 1.2 - validate fedavg
    bash validate_fed.sh

    # step 2 - train 
    for c in 1 2 3 4; do
        python ../../train_baselines.py cuda:${cudalist[$c-1]} $c "False" "True" "../../data_configs/camvid.yml" $num_center_epochs "." &
    done
    #wait for all centers to train
    wait
done