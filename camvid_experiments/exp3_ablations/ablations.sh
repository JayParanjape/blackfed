#!/bin/bash
for clients in 10 20 30 40 50 
do
    for servers in 10 20 30 40 50 
    do
        if [[ $clients == 10 && $servers == 10 ]] 
        then 
            continue 
        fi
        if [[ $clients == 10 && $servers == 20 ]] 
        then 
            continue 
        fi
        if [[ $clients == 10 && $servers == 30 ]] 
        then 
            continue 
        fi

        echo client $clients server $servers
        cp -r saved_models_rev_client${clients}_server${servers} safe
        python driver_rev.py cuda:4 $clients $servers >> testing_progress_rev_client${clients}_server${servers}.txt

    done
done