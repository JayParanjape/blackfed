#!/bin/bash
echo "FedAvg Algorithm for cityscapes Dataset"
num_fed_epochs=10
num_center_epochs=25
cudalist=(1 2 3 4 5 6)

#initial training
# for c in 1 2 3 4 5 6; do
#         python ../../train_baselines.py cuda:${cudalist[$c-1]} $(( $c + 1 - 1 )) "False" "False" "../../data_configs/cityscapes.yml" $num_center_epochs "." &
#     done
# wait
# for c in 1 2 3 4 5 6; do
#         python ../../train_baselines.py cuda:${cudalist[$c-1]} $(( $c + 6 )) "False" "False" "../../data_configs/cityscapes.yml" $num_center_epochs "." &
#     done
# #wait for all centers to train
# wait
# for c in 1 2 3 4 5 6; do
#         python ../../train_baselines.py cuda:${cudalist[$c-1]} $(( $c + 12 )) "False" "False" "../../data_configs/cityscapes.yml" $num_center_epochs "." &
#     done
# #wait for all centers to train
# wait

for i in $(seq 1 $num_fed_epochs); do
    #step 1 - perform fedavg
    python ../../fed_avg.py "fedavg"
    #step 1.2 - validate fedavg
    bash validate_fed.sh

    # step 2 - train 
    for c in 1 2 3 4 5 6; do
        python ../../train_baselines.py cuda:${cudalist[$c-1]} $c "False" "True" "../../data_configs/cityscapes.yml" $num_center_epochs "." &
    done
    wait
    for c in 1 2 3 4 5 6; do
            python ../../train_baselines.py cuda:${cudalist[$c-1]} $(( $c + 6 )) "False" "True" "../../data_configs/cityscapes.yml" $num_center_epochs "." &
        done
    #wait for all centers to train
    wait
    for c in 1 2 3 4 5 6; do
            python ../../train_baselines.py cuda:${cudalist[$c-1]} $(( $c + 12 )) "False" "True" "../../data_configs/cityscapes.yml" $num_center_epochs "." &
        done
    #wait for all centers to train
    wait
done