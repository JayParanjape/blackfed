#!/bin/bash
echo "FedAvg Algorithm for Polypgen Dataset"
num_fed_epochs=10
num_center_epochs=50
cudalist=(0 2 4 5 7 7)

#initial training
for c in 1 2 3 4 5 6; do
        python ../../train_baselines.py cuda:${cudalist[$c-1]} $c "False" "False" "../../data_configs/polypgen.yml" $num_center_epochs "." &
    done
#wait for all centers to train
wait

for i in $(seq 1 $num_fed_epochs); do
    #step 1 - perform fedavg
    python ../../fed_avg.py "fedavg"
    #step 1.2 - validate fedavg
    bash validate_fed.sh

    # step 2 - train 
    for c in 1 2 3 4 5 6; do
        python ../../train_baselines.py cuda:${cudalist[$c-1]} $c "False" "True" "../../data_configs/polypgen.yml" $num_center_epochs "." &
    done
    #wait for all centers to train
    wait
done